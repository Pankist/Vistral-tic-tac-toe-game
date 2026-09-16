# AWS demo topology — as deployed

Account PVLX (415144300182), region us-east-1. Everything lives in a
dedicated VPC so the whole system tears down cleanly. Deploy is git-based:
the instance bootstrap clones this repo, runs `make setup` **and `make test`
— a red suite never becomes a running service** — then installs the systemd
unit. The OpenRouter key never touches git: it sits in SSM Parameter Store
and the instance role is allowed to read exactly that one parameter.

Ubuntu 24.04 note: there is no `awscli` apt package on noble — the bootstrap
uses `snap install aws-cli --classic` to fetch the SSM parameter.

| Piece | Value |
|---|---|
| VPC | `vpc-049251e257c235cbd` (10.42.0.0/16, `vistral-ttt`) |
| Subnets | `subnet-02f34e5761ac29ef6` (1a), `subnet-008e38320929e091d` (1b) |
| EC2 | `i-05af14deb138b9831` — t3.small, Ubuntu 24.04, SSM-managed (no SSH keypair) |
| SG (app) | `sg-0ac3cf838fc584464` — :8000 from the ALB SG only |
| ALB | `vistral-ttt-1566484477.us-east-1.elb.amazonaws.com` — HTTP :80, **idle timeout 600s** (default 60s kills long-lived WebSockets), health `GET /health` |
| Secret | SSM `SecureString` `/vistral-ttt/openrouter-api-key` |
| IAM role | `vistral-ttt-ec2`: `AmazonSSMManagedInstanceCore` + `ssm:GetParameter` on `parameter/vistral-ttt/*` |

## Demo

Client stays local (camera needs a secure context; localhost qualifies):

```bash
make client        # serves client/ on http://localhost:3000 (CLIENT_PORT=3001 if taken)
open http://localhost:3000
```

The ALB is the client's built-in default backend — no query string needed;
`?backend=ws://...` overrides it for any other host.

Fallback is one param: drop `?backend=` and run `make dev` to flip the whole
demo to localhost if the venue network misbehaves.

## Operations (all via SSM, no SSH)

```bash
# redeploy after a push
aws ssm send-command --region us-east-1 --instance-ids i-05af14deb138b9831 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["cd /home/ubuntu/app && sudo -u ubuntu git pull && sudo -u ubuntu make test && systemctl restart ttt-agent"]'

# logs
aws ssm send-command --region us-east-1 --instance-ids i-05af14deb138b9831 \
  --document-name AWS-RunShellScript \
  --parameters 'commands=["journalctl -u ttt-agent -n 50 --no-pager"]'

# interactive shell
aws ssm start-session --region us-east-1 --target i-05af14deb138b9831
```

## Verify

```bash
curl http://vistral-ttt-1566484477.us-east-1.elb.amazonaws.com/health
.venv/bin/python scripts/ws_smoke.py ws://vistral-ttt-1566484477.us-east-1.elb.amazonaws.com/ws
```
