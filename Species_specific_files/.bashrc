alias interactive_gpu="srun -p scavenger-gpu -n 10 --pty --mem 60000 --gres=gpu:1 --cpu-bind=no -t 0-30:00 /bin/bash"
