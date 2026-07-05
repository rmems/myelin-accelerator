# Commit signing

This repository requires all commits to be SSH-signed and verified by
GitHub. Each contributor signs with **their own local SSH key** — not
the maintainer's. GitHub verifies the signature against any signing
key registered on the contributor's GitHub account.

## Configuration (one-time, on each dev machine)

1. **Have an SSH key** (any modern ed25519 or RSA works):

   ```bash
   ssh-keygen -t ed25519 -C "<your-github-email>"
   ```

2. **Register it on GitHub** as both an auth key *and* a signing key.
   GitHub treats these as two separate lists — uploading only an auth
   key does not enable commit signing.

   - `https://github.com/settings/keys` → **New SSH key** (authentication)
   - `https://github.com/settings/keys` → **New SSH key** *(Signing key
     is the same form, but check the "Authentication & Signing" toggle
     to mark it as a signing key)*. Alternatively, the
     [GitHub CLI](https://cli.github.com/) can do it:
     `gh ssh-key add ~/.ssh/id_ed25519.pub --type signing`

3. **Tell git to sign with SSH** using your local key:

   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519
   ```

4. **(Optional) Enable local verification** so `git verify-commit` and
   `git log --show-signature` work without errors. The `allowed_signers`
   file maps `<email> -> <ssh public key>` for each signatory.

   ```bash
   mkdir -p ~/.config/git
   printf '%s %s\n' "<your-verified-github-email>" \
     "$(cat ~/.ssh/id_ed25519.pub)" > ~/.config/git/allowed_signers
   git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
   chmod 600 ~/.config/git/allowed_signers
   ```

## Why SSH and not GPG

GitHub verifies a commit signature by looking up the committer email
in its user database and then checking the signature against any
signing keys registered for that account. SSH signing reuses the
same `id_ed25519` key already uploaded for SSH access; GPG signing
requires uploading a separate GPG public key to GitHub. SSH is
simpler and avoids the extra key.
