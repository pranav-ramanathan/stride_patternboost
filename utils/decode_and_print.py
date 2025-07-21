import argparse
import os
import torch

def get_parser():
    """Sets up the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Decode construction strings from a file and print their grid representations.")
    parser.add_argument("--input_file", "-i", type=str, required=True, help="Path to the file containing construction strings.")
    parser.add_argument("--grid_size", "-N", type=int, required=True, help="The grid size (e.g., 11).")
    parser.add_argument("--output_file", "-o", type=str, default=None, help="Optional path to an output file to save the decoded grids.")
    return parser

def write_output(line, file_handle=None):
    """Prints to console and writes to file handle if provided."""
    print(line)
    if file_handle:
        file_handle.write(line + '\n')

def print_grid(construction_grid, score, title="Construction", file_handle=None):
    """Prints a 2D text representation of a single construction."""
    grid_list = construction_grid.tolist()
    N = len(grid_list)
    
    write_output(f"\n--- {title} (Score: {score}) ---", file_handle)
    header = "  " + " ".join(map(str, range(N)))
    write_output(header, file_handle)
    write_output("  " + "-" * (2 * N -1), file_handle)

    for r in range(N):
        row_str = [str(r) + "|"]
        for c in range(N):
            if grid_list[r][c] == 1:
                row_str.append('X')
            else:
                row_str.append('.')
        write_output(' '.join(row_str), file_handle)
    write_output("-" * (2 * N + 3), file_handle)

def main():
    """Main execution function."""
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        exit(1)

    output_file_handle = None
    if args.output_file:
        try:
            # Ensure the directory exists
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_file_handle = open(args.output_file, 'w')
        except IOError as e:
            print(f"Error: Could not open output file {args.output_file} for writing: {e}")
            exit(1)

    N = args.grid_size
    
    with open(args.input_file, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                tokens = [int(t[1:]) for t in line.split(',') if t]
                score = len(tokens)
                
                grid = torch.zeros((N, N), dtype=torch.int8)

                for token_num in tokens:
                    row = token_num // N
                    col = token_num % N
                    if 0 <= row < N and 0 <= col < N:
                        grid[row, col] = 1
                    else:
                        print(f"Warning: Token {token_num} is out of bounds for grid size {N} on line {i+1}.")

                print_grid(grid, score, title=f"Construction #{i+1}", file_handle=output_file_handle)

            except ValueError as e:
                print(f"Error processing line {i+1}: '{line}'")
                print(f"  -> {e}")
            except Exception as e:
                print(f"An unexpected error occurred on line {i+1}: {e}")
    
    if output_file_handle:
        print(f"\nDecoded grids also saved to {args.output_file}")
        output_file_handle.close()

    print("\nDecoding complete.")

if __name__ == "__main__":
    main() 