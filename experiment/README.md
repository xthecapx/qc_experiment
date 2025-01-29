# MIT-iQuHack Jupyter Environment

This repository contains a Docker environment for running Jupyter notebooks.

## Requirements

- Docker
- Docker Compose

## Running the Environment

1. Start the Jupyter environment:

```bash
docker-compose -f docker-compose.yml up --build -d
```

2. Get the Jupyter access URL:
   - Check the container logs for a URL that looks like: `http://127.0.0.1:4321/lab?token=<your-token>`
   - You can find this by either:
     ```bash
     docker-compose logs | grep token
     ```
     or by viewing all logs:
     ```bash
     docker-compose logs
     ```

3. Open the URL in your web browser to access Jupyter Lab

4. Optionally, you can use the `startup.sh` script to automatically search for the URL:

```bash
chmod +x startup.sh
./startup.sh
```

## Contributing

### Development Workflow

This environment supports hot-reloading of Python modules, allowing you to:
1. Edit Python files directly
2. See changes reflect immediately in Jupyter notebooks without restarting

To enable auto-reloading in your notebook:
1. Add these lines at the beginning of your Jupyter notebook:
   ```python
   %load_ext autoreload
   %autoreload 2
   ```
2. Now you can edit any .py files and the changes will be automatically picked up when you run notebook cells

### Best Practices
- Keep implementation code in .py files
- Use notebooks primarily for visualization and experimentation
- Save your notebooks frequently
- Commit both .py files and notebooks to version control

## Stopping the Environment

To stop and remove the containers:

```bash
docker-compose -f docker-compose.yml down
```

## Access

The Jupyter environment will be available at `http://localhost:4321/lab?token=<your-token>` once started. You'll need the token from the container logs to access it for the first time.