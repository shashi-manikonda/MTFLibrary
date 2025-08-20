import os
import glob
import sys

BACKENDS_DIR = "src/mtflib/backends"

def find_extensions():
    """Find all compiled extension files, whether enabled or disabled."""
    enabled = glob.glob(f"{BACKENDS_DIR}/**/*.so", recursive=True)
    disabled = glob.glob(f"{BACKENDS_DIR}/**/*.so.disabled", recursive=True)
    return enabled + disabled

def switch_backend(enable=True):
    """
    Enable or disable the compiled backends by renaming them.

    :param enable: If True, ensures extensions have '.so' suffix.
                   If False, renames them to '.so.disabled'.
    """
    extensions = find_extensions()
    if not extensions:
        print("No compiled extensions found.")
        return

    for ext_path in extensions:
        if enable:
            # We want to enable the backend, so rename .so.disabled to .so
            if ext_path.endswith('.so.disabled'):
                new_path = ext_path.replace('.so.disabled', '.so')
                print(f"Enabling {ext_path} -> {new_path}...")
                os.rename(ext_path, new_path)
            elif ext_path.endswith('.so'):
                print(f"Backend {ext_path} is already enabled.")
        else:
            # We want to disable the backend, so rename .so to .so.disabled
            if ext_path.endswith('.so'):
                new_path = ext_path + '.disabled'
                print(f"Disabling {ext_path} -> {new_path}...")
                os.rename(ext_path, new_path)
            elif ext_path.endswith('.so.disabled'):
                print(f"Backend {ext_path} is already disabled.")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['enable', 'disable']:
        print("Usage: python manage_backends.py [enable|disable]")
        sys.exit(1)

    action = sys.argv[1]

    if action == 'enable':
        switch_backend(enable=True)
    else:
        switch_backend(enable=False)

    print("Done.")
