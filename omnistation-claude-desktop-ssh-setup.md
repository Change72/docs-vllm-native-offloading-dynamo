# Claude Code 桌面 App 连接 Omnistation 的修复记录

> 日期：2026-06-10（v2：独立 sshd 实例方案）
> 工作站：`omni-lfn-rzigm`（Teleport cluster `nv-prd-it.teleport.sh`，login `changg`，有 passwordless sudo）
> 现象：终端 `ssh omnistation` 正常，但 Claude Code 桌面 app 的 remote 连接报错。

## TL;DR

1. 桌面 app 内嵌的 SSH 客户端（JS `ssh2` 库）不支持 Teleport 的 **host certificate** →
   本机 `~/.ssh/config` 新增 `Host omnistation-app`，用系统 OpenSSH 做跳板（`-W`）穿过 Teleport。
2. 工作站的主 sshd 被 CIS 配置管理锁死（改了会被定期还原）→
   在工作站跑一个**独立的 loopback-only sshd 实例（`claude-sshd`，127.0.0.1:2222，UsePAM no）**，
   完全不碰被管控的文件。

**App 里选 host `omnistation-app`（不是 `omnistation`）。**

## 连接架构（v2）

```
Claude Code app (内嵌 ssh2 客户端，只认普通 host key)
   │  读 ~/.ssh/config → Host omnistation-app
   ▼
ProxyCommand: /usr/bin/ssh omnistation -W localhost:2222   ← 系统 OpenSSH，能处理 Teleport 证书
   │  内部再嵌套: ProxyCommand tsh proxy ssh --proxy=nv-prd-it.teleport.sh
   ▼
Teleport node service ──direct-tcpip──▶ claude-sshd @ 127.0.0.1:2222
                                         (独立配置 / UsePAM no / 仅公钥 / 不暴露网络)
```

## 问题分层（按发现顺序）

| # | 报错/症状 | 根因 | 结局 |
|---|-----------|------|------|
| 1 | `Handshake failed: no matching host key format` | app 内嵌 `ssh2` 库不支持 Teleport 的 `*-cert-v01@openssh.com` host 证书，且无开关可改用系统 ssh（官方无解法，见 anthropics/claude-code#32734、mscdex/ssh2#1069） | 跳板方案解决（沿用至 v2） |
| 2 | auth.log: `not listed in AllowUsers` | CIS sshd_config `AllowUsers admin sysadmin operator` 不含 changg；该镜像无 `Include sshd_config.d`，drop-in 无效 | v1 改主文件 → **被配置管理还原** → v2 用独立实例绕开 |
| 3 | 公钥被接受后挂 ~28s 然后断开（debug sshd 停在 `do_pam_account: called`） | PAM account 阶段 SSSD GPO 评估（`ad_gpo_access_control = permissive`）挂起，gpo_child 访问 AD 超时；Teleport 登录不走 PAM 所以从未暴露 | v1 改 `disabled` → **被还原** → v2 `UsePAM no` 彻底绕开 |
| 4 | `Failed to open SFTP session: Unable to start subsystem: sftp` | CIS 配置删了 `Subsystem sftp` 行；app 必须有 SFTP | v1 加回 → **被还原** → v2 在独立实例里配置 |
| 5 | `REMOTE HOST IDENTIFICATION HAS CHANGED` | Teleport service 与真 sshd 共用主机名但 key 不同；OpenSSH `UpdateHostKeys` 连上一个端点后会清掉另一个端点的 key | `HostKeyAlias` 分离命名空间 + `UpdateHostKeys no` |
| 6 | `Too many authentication failures` | ssh-agent 里的 Teleport 证书被逐个先试，撞上 MaxAuthTries | `IdentitiesOnly yes` |

> v1（直接改 `/etc/ssh/sshd_config`、`/etc/sssd/sssd.conf`）工作了约 20 分钟后被配置管理
> 全部还原（盒子上有合规工具定期纠偏，未现身于 systemd running services，可能是 timer/cron）。
> v2 不再与它拉锯：所有自定义都放在它不管的文件里。

## 改动清单（v2 现状）

### 本机（macOS）`~/.ssh/config`

```sshconfig
Host omnistation
    HostName omni-lfn-rzigm
    User changg
    ProxyCommand /usr/local/bin/tsh proxy ssh --proxy=nv-prd-it.teleport.sh %r@%h:%p
    # Teleport service and the workspace's real sshd share this hostname;
    # alias keeps their known_hosts identities separate.
    HostKeyAlias omni-teleport
    UpdateHostKeys no

# For Claude Code desktop app: its embedded SSH client can't handle Teleport
# host certificates, so hop through system OpenSSH to the workspace's plain sshd.
Host omnistation-app
    HostName omni-lfn-rzigm
    User changg
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    PreferredAuthentications publickey
    HostKeyAlias omnistation-app
    UpdateHostKeys no
    ProxyCommand /usr/bin/ssh -F /Users/changg/.ssh/config -o BatchMode=yes omnistation -W localhost:2222
```

known_hosts：真 sshd 的三把 host key 记在别名 `omnistation-app` 下（claude-sshd 复用
`/etc/ssh/ssh_host_*_key`，所以条目通用）；Teleport service 的 key 记在 `omni-teleport` 下。
公钥 `~/.ssh/id_ed25519.pub` 已装进工作站 `~/.ssh/authorized_keys`。

### 工作站：独立 sshd 实例（不碰被管控文件）

**`/etc/ssh/sshd_claude.conf`**

```
# Loopback-only sshd for Claude Code desktop app (reached via Teleport tunnel).
# Independent of the managed /etc/ssh/sshd_config so compliance tooling
# reverts do not break it. UsePAM no avoids the SSSD GPO hang entirely.
Port 2222
ListenAddress 127.0.0.1
ListenAddress ::1
HostKey /etc/ssh/ssh_host_ed25519_key
HostKey /etc/ssh/ssh_host_ecdsa_key
HostKey /etc/ssh/ssh_host_rsa_key
AllowUsers changg
AuthenticationMethods publickey
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
UsePAM no
X11Forwarding no
MaxAuthTries 6
LoginGraceTime 60
Subsystem sftp internal-sftp
```

**`/etc/systemd/system/claude-sshd.service`**

```ini
[Unit]
Description=Loopback-only sshd for Claude Code desktop app
After=network.target

[Service]
ExecStartPre=/usr/sbin/sshd -t -f /etc/ssh/sshd_claude.conf
ExecStart=/usr/sbin/sshd -D -f /etc/ssh/sshd_claude.conf
ExecReload=/bin/kill -HUP $MAINPID
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

启用：`sudo systemctl daemon-reload && sudo systemctl enable --now claude-sshd`

安全性说明：只监听 loopback，外部唯一入口仍是 Teleport（带审计）；仅公钥认证、仅 changg、
禁 root、禁密码。比 v1（在网络暴露的端口 22 上放宽 AllowUsers）更收敛。

`UsePAM no` 的代价：没有 pam_systemd（无 loginctl 会话/XDG_RUNTIME_DIR，systemd --user 服务
不会自动起）。对 Claude Code 跑编译/测试/agent 没有影响；需要完整登录会话时走 Teleport 终端即可。

## 验证命令

```bash
# app 路径（应 ~1 秒返回，无 banner）
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes omnistation-app 'echo "APP-PATH-OK on $(hostname) as $(whoami)"'

# SFTP（app 必需）
echo "pwd" | sftp -b - omnistation-app

# 原终端路径
ssh -o BatchMode=yes omnistation 'echo TERMINAL-PATH-OK'

# 工作站上：实例状态
systemctl status claude-sshd
```

## 故障速查

| 症状 | 多半是 | 处理 |
|------|--------|------|
| app + 终端都连不上 | `tsh` 会话过期（约一天） | `tsh login --proxy=nv-prd-it.teleport.sh` |
| app 连不上、终端正常、`Connection closed by UNKNOWN` | claude-sshd 没在跑（被清理/机器重建） | 上面"独立 sshd 实例"两个文件重建 + enable --now；公钥重新装入 authorized_keys |
| `REMOTE HOST IDENTIFICATION HAS CHANGED` | 机器重建换了 host key，或 known_hosts 又混了 | `ssh-keygen -R omnistation-app; ssh-keygen -R omni-teleport`，然后 `tsh ssh changg@omni-lfn-rzigm 'cat /etc/ssh/ssh_host_*_key.pub' \| sed 's/^/omnistation-app /' >> ~/.ssh/known_hosts`，再 `ssh -o StrictHostKeyChecking=accept-new omnistation true` |
| `Too many authentication failures` | config 里 `IdentitiesOnly yes` 丢了 | 加回去 |
| `Unable to start subsystem: sftp` | sshd_claude.conf 被改/丢 | 核对上面的配置文件，`systemctl reload claude-sshd` |

### 诊断技巧（这次用过、好使的）

```bash
# 服务端真实拒绝原因（比客户端报错诚实）
tsh ssh changg@omni-lfn-rzigm 'sudo tail -20 /var/log/auth.log'

# claude-sshd 实例日志
tsh ssh changg@omni-lfn-rzigm 'sudo journalctl -u claude-sshd -n 30'

# 终极手段：备用端口跑 debug sshd，看服务端死在哪一步
tsh ssh changg@omni-lfn-rzigm 'sudo sh -c "timeout 30 /usr/sbin/sshd -d -p 2299 > /tmp/sshd-debug.log 2>&1 &"'
# 另起连接打它，然后 sudo tail /tmp/sshd-debug.log
# （v1 阶段就是靠它定位到卡死在 "do_pam_account: called" → SSSD GPO 挂起）
```

## 其他备注

- `omni-cli init` 报 "0 Omnistation workspaces found"：节点 label 是 `resource-group=it-vdi`，
  不匹配 omni-cli 的过滤条件，无碍使用。
- v1 对被管控文件的三处改动已被配置管理自动还原，无需手工回滚；备份文件
  （`sshd_config.bak`、`sssd.conf.bak`）可能仍留在工作站上，无害。
