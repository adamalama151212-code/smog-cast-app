# Source before any `docker compose` command that should reach the warehouse (Git Bash):  . tools/dev-env.sh
# Every compose command re-reads these variables; a shell without them would recreate the app without a token.
# The token is short-lived (about 1 h) and stays in this shell's memory only – never in a file.
#
# Which workspace, which warehouse and which Databricks CLI profile are local settings: they live in .env next to
# docker-compose.yml (outside git, see .env.example). A value already set in the shell wins.
_local() { sed -n "s/^$1=//p" "$(dirname "${BASH_SOURCE[0]}")/../.env" 2>/dev/null | tr -d '\r' | tail -n 1; }
PROFILE="${DATABRICKS_PROFILE:-$(_local DATABRICKS_PROFILE)}"
export DATABRICKS_HOST="${DATABRICKS_HOST:-$(_local DATABRICKS_HOST)}"
export DATABRICKS_WAREHOUSE_ID="${DATABRICKS_WAREHOUSE_ID:-$(_local DATABRICKS_WAREHOUSE_ID)}"
unset -f _local
if [ -z "$PROFILE" ] || [ -z "$DATABRICKS_HOST" ] || [ -z "$DATABRICKS_WAREHOUSE_ID" ]; then
  echo "tools/dev-env.sh: set DATABRICKS_PROFILE, DATABRICKS_HOST and DATABRICKS_WAREHOUSE_ID in .env (see .env.example)" >&2
else
  DATABRICKS_TOKEN="$(databricks auth token --profile "$PROFILE" -o json | python -c "import json,sys; print(json.load(sys.stdin)['access_token'])")"
  export DATABRICKS_TOKEN
fi
