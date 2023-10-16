"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """ST."""


if __name__ == "__main__":
    main(prog_name="starling-tool")  # pragma: no cover
