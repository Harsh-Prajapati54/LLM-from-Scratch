"""
Hugging Face Model Manager
--------------------------
One-file terminal utility for uploading/downloading model files
(or complete model folders) to/from Hugging Face.

First run:
    pip install -U huggingface_hub

Then:
    python hf_model_manager.py
"""

from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, login, whoami


def ensure_login():
    """Check Hugging Face authentication and ask for login if needed."""
    try:
        info = whoami()
        username = info.get("name", "unknown")
        print(f"\nLogged in as: {username}")
        return username
    except Exception:
        print("\nYou are not logged in to Hugging Face.")
        print("A Hugging Face token with the required permissions will be needed.")
        try:
            login()
            info = whoami()
            username = info.get("name", "unknown")
            print(f"Logged in as: {username}")
            return username
        except Exception as exc:
            print(f"\nLogin failed: {exc}")
            sys.exit(1)


def ask_nonempty(prompt: str) -> str:
    while True:
        value = input(prompt).strip().strip('"')
        if value:
            return value
        print("Please enter a value.")


def upload():
    print("\n========== UPLOAD MODEL ==========")
    username = ensure_login()

    source = Path(
        ask_nonempty(
            "Enter local model FILE or FOLDER path:\n> "
        )
    )

    if not source.exists():
        print(f"\nPath does not exist:\n{source}")
        return

    repo_name = ask_nonempty(
        "\nEnter Hugging Face repo name "
        "(example: my-llm-v2):\n> "
    )

    repo_id = f"{username}/{repo_name}"

    private_choice = input(
        "\nMake repository private? [Y/n]: "
    ).strip().lower()
    private = private_choice != "n"

    api = HfApi()

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )

        print(f"\nRepository ready: https://huggingface.co/{repo_id}")

        if source.is_file():
            remote_name = input(
                f"\nRemote filename [{source.name}]: "
            ).strip()

            if not remote_name:
                remote_name = source.name

            print("\nUploading file...")
            api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model",
            )
        else:
            print("\nUploading folder...")
            api.upload_folder(
                folder_path=str(source),
                repo_id=repo_id,
                repo_type="model",
            )

        print("\n✅ UPLOAD COMPLETE")
        print(f"Repo: https://huggingface.co/{repo_id}")

    except Exception as exc:
        print(f"\n❌ Upload failed:\n{exc}")


def download():
    print("\n========== DOWNLOAD MODEL ==========")
    ensure_login()

    repo_id = ask_nonempty(
        "\nEnter Hugging Face repo ID "
        "(example: harsh-prajapati54/my-llm-v2):\n> "
    )

    destination = Path(
        ask_nonempty(
            "\nEnter LOCAL destination folder:\n"
            r"Example: C:\LLM from Scratch\output\models"
            "\n> "
        )
    )

    destination.mkdir(parents=True, exist_ok=True)

    choice = input(
        "\nDownload:\n"
        "1. One file\n"
        "2. Entire repository\n"
        "Choose [1/2]: "
    ).strip()

    try:
        if choice == "1":
            filename = ask_nonempty(
                "\nEnter filename in the repo "
                "(example: model.pth):\n> "
            )

            print("\nDownloading file...")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(destination),
            )

            print("\n✅ DOWNLOAD COMPLETE")
            print(f"Saved to: {path}")

        elif choice == "2":
            print("\nDownloading entire repository...")
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(destination),
            )

            print("\n✅ DOWNLOAD COMPLETE")
            print(f"Saved to: {path}")

        else:
            print("\nInvalid choice. Please choose 1 or 2.")

    except Exception as exc:
        print(f"\n❌ Download failed:\n{exc}")


def main():
    print("\n========================================")
    print("      HUGGING FACE MODEL MANAGER")
    print("========================================")

    while True:
        print(
            "\nWhat do you want to do?\n"
            "1. Upload model\n"
            "2. Download model\n"
            "3. Exit"
        )

        choice = input("\nChoose [1/2/3]: ").strip()

        if choice == "1":
            upload()
        elif choice == "2":
            download()
        elif choice == "3":
            print("\nBye!")
            break
        else:
            print("\nInvalid choice.")

        again = input(
            "\nDo you want to do another operation? [Y/n]: "
        ).strip().lower()

        if again == "n":
            print("\nDone!")
            break


if __name__ == "__main__":
    main()
