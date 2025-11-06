# PostgreSQL Cloud Setup

## Why Postgres?

When training on multiple machines (cluster + local), PostgreSQL provides a **single source of truth** without the hassle of syncing SQLite databases.

✅ **Benefits:**
- One database for all machines
- No syncing or merging needed  
- Real-time updates from anywhere
- Better for concurrent access
- Works seamlessly with existing code

## Quick Setup Options

### Option 1: Supabase (Free, Easiest)

1. Go to [https://supabase.com](https://supabase.com) and sign up
2. Create a new project
3. Get your connection string from **Settings > Database**
4. Add to your environment:

```bash
export DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT].supabase.co:5432/postgres"
```

### Option 2: Railway (Free tier)

1. Go to [https://railway.app](https://railway.app) and sign up
2. Create a new project > Add PostgreSQL
3. Copy the `DATABASE_URL` from the **Connect** tab
4. Add to your environment

### Option 3: Neon (Serverless, Free)

1. Go to [https://neon.tech](https://neon.tech) and sign up
2. Create a new project
3. Copy the connection string
4. Add to your environment

### Option 4: Your own server

If you have access to a Postgres server:

```bash
export DATABASE_URL="postgresql://username:password@hostname:5432/database_name"
```

## Usage

### On Your Local Machine

```bash
# Add to ~/.bashrc or ~/.zshrc
export DATABASE_URL="postgresql://postgres:yourpass@db.xxxxx.supabase.co:5432/postgres"

# Then train as usual
cd projects/project2
uv run train.py model.latent_dim=16
```

### On the Cluster

```bash
# Add to ~/.bashrc or submit script
export DATABASE_URL="postgresql://postgres:yourpass@db.xxxxx.supabase.co:5432/postgres"

# Train - no need for db_path override!
uv run train.py model.latent_dim=16 training.epochs=100
```

### Verify Connection

```bash
# Test database connection
uv run python -c "from infrastructure.database import ExperimentDatabase; db = ExperimentDatabase(); print('Connected!'); db.close()"
```

**Expected output:**
```
✅ Connected to PostgreSQL database
Connected!
```

## Fallback to SQLite

If `DATABASE_URL` is not set, the code automatically falls back to SQLite:

```bash
unset DATABASE_URL
uv run train.py model.latent_dim=16
# ✅ Connected to SQLite database: experiments/runs.db
```

## Security Best Practices

**❌ DON'T:**
- Commit `DATABASE_URL` to git
- Share connection strings publicly
- Use same password for production and dev

**✅ DO:**
- Use environment variables
- Rotate credentials periodically
- Use read-only credentials for analysis scripts (if possible)
- Consider IP allowlists on production databases

## Migrating from SQLite

If you have existing SQLite data you want to preserve:

```bash
# 1. Export SQLite to CSV
uv run python -c "
from infrastructure.database import ExperimentDatabase
import pandas as pd
db = ExperimentDatabase('experiments/runs.db')
df = db.get_all_experiments()
df.to_csv('experiments_backup.csv', index=False)
db.close()
"

# 2. Set up Postgres (DATABASE_URL)
export DATABASE_URL="postgresql://..."

# 3. Import CSV to Postgres (manually or with a script)
# Note: The tables will auto-create on first use
```

## Troubleshooting

### "Could not connect to server"
- Check your internet connection
- Verify `DATABASE_URL` is correct
- Check if your IP is allowed (some services require allowlisting)

### "psycopg2 not installed"
```bash
uv add psycopg2-binary
```

### "SSL connection required"
Add `?sslmode=require` to your connection string:
```bash
export DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require"
```

### Want to switch back to SQLite temporarily?
```bash
unset DATABASE_URL
# Now uses local SQLite
```

## Performance Tips

- **Connection pooling**: For production, consider using connection pooling (pgbouncer)
- **Indexes**: The default schema includes indexes on `run_name` (unique constraint)
- **Backups**: Most cloud providers auto-backup. Check your provider's settings
- **Cost**: Free tiers should be sufficient for this project (~1-10 MB total data)

## FAQ

**Q: Will this slow down training?**  
A: No. Database writes happen between epochs (few seconds), not during training.

**Q: What if the internet goes down mid-training?**  
A: Training continues. Database writes will fail, but checkpoints are saved locally. You can manually update the DB later.

**Q: Can multiple machines train simultaneously?**  
A: Yes! Each needs a unique `experiment.run_name` to avoid conflicts.

**Q: How do I see my data?**  
A: Use the analysis scripts as usual:
```bash
uv run python analyze.py
uv run python generate_report.py
```

Or connect with a GUI tool like:
- [Supabase Studio](https://supabase.com) (built-in)
- [DBeaver](https://dbeaver.io)
- [pgAdmin](https://www.pgadmin.org)

