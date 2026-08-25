# platform/ — self-host IaC (Q13)

The `platform` profile's infrastructure so a **user** can stand up the live
streaming ingestion stack on **their own** hardware — never the operator's.

Planned (TODO, Phase 2/3):
- `docker-compose.yml` — Redpanda (Kafka API) + a worker + DuckDB, one command.
- `helm/` — Kubernetes chart for the same (Redpanda + Spark + workers).
- `ansible/` — playbook to provision it on a bare VM.

The `lite` profile needs **none** of this — it runs embedded with DuckDB. This
directory is only for users who want continuous public-corpus ingestion at scale.
