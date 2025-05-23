import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--term", type=str, help="The common term to search on the files", required=True)
args = parser.parse_args()

to_remove = [file for file in os.listdir("logs") if args.term in file]
print(to_remove)
for file in to_remove:
    os.remove(os.path.join("logs", file))

print("Files removed")
