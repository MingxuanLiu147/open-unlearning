#!/bin/bash
set -e
YAML_FILE="${1:?用法: bash scripts/pai/submit.sh <yaml文件>}"
[ ! -f "$YAML_FILE" ] && echo "错误: $YAML_FILE 不存在" && exit 1
export YAML_FILE
exec python3 -c "
import urllib.request, urllib.error, json, ssl, sys, os, yaml

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
PAI = 'https://210.75.240.150'

# Login
data = json.dumps({'username':'liumingxuan','password':'liumingxuan123.','expiration':604800}).encode()
req = urllib.request.Request(f'{PAI}/rest-server/api/v2/authn/basic/login',
    data=data, headers={'Content-Type':'application/json'}, method='POST')
try:
    token = json.loads(urllib.request.urlopen(req, context=ctx).read())['token']
    print('登录成功')
except Exception as e:
    print(f'登录失败: {e}'); sys.exit(1)

# Read YAML
yaml_file = os.environ['YAML_FILE']
with open(yaml_file) as f:
    yaml_content = f.read()
job_name = yaml.safe_load(yaml_content).get('name', 'unknown')
print(f'作业名: {job_name}')

# Submit as text/yaml
print('提交中...')
req = urllib.request.Request(f'{PAI}/rest-server/api/v2/jobs',
    data=yaml_content.encode('utf-8'),
    headers={'Content-Type':'text/yaml','Authorization':f'Bearer {token}'},
    method='POST')
try:
    resp = urllib.request.urlopen(req, context=ctx)
    print(f'提交成功! HTTP {resp.status}')
    print(f'查看: {PAI}/job-detail.html?username=liumingxuan&jobName={job_name}')
except urllib.error.HTTPError as e:
    print(f'提交失败: HTTP {e.code}')
    print(e.read().decode()[:500]); sys.exit(1)
"
