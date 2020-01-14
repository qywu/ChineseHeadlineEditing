import os
import subprocess


alpha_beta = [
              (0.5, 20.0),
              (0.1, 30.0),
              (1.0, 20.0),
              (2.0, 20.0),
              ]

if __name__ == "__main__":
    for a, b in alpha_beta:
        output_dir = f"results_{a}_{b}"
        subprocess.run(["python", "main.py", f"-a {a}", f"-b {b}", f"-o {output_dir}"])