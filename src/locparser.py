import argparse


def parser():
    parser = argparse.ArgumentParser(
        description="prediction of the interactions for RING software"
    )
    # Add the command-line arguments
    parser.add_argument("-u", "--update_dssp", action="store_true", help="Update DSSP")
    parser.add_argument(
        "-g", "--generate_dssp", action="store_true", help="Generate DSSP"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "-t", "--multithreading", action="store_true", help="Enable multithreading"
    )
    # parser.add_argument("-m", "--model", help="model name", required=True)
    # parser.add_argument(
    #     "-n", "--normalization", help="normalization name", required=True
    # )
    # parser.add_argument(
    #     "-d", "--manipulation", help="dataset manipulation type", required=True
    # )

    args = parser.parse_args()

    # if not args.model:
    #     print("Error: Model name is required.")
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)

    # if not args.normalization:
    #     print("Error: Normalization name is required.")
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)

    # if not args.manipulation:
    #     print("Error: Dataset manipulation type is required.")
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    if not args.update_dssp:
        args.update_dssp = False
    if not args.generate_dssp:
        args.generate_dssp = False
    if not args.debug:
        args.debug = False
    return args
