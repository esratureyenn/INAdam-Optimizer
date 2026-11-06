import os

def save_args(args, directory, filename='parameters.txt'):
  filename = os.path.join(directory, filename)
  with open(filename, 'w') as f:
      for arg, value in vars(args).items():
          f.write(f"{arg}: {value}\n")
  print(f"Arguments saved to {filename}")