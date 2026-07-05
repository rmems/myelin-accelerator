# Commit signing

This repository uses SSH commit signing via the maintainer's
`id_ed25519` key (registered on GitHub as "Ship of Theseus: Fedora").

## Configuration (one-time, on each dev machine)

```bash
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519

# Optional: enable local verification of SSH signatures.
# This is what makes `git verify-commit` and `git log --show-signature`
# work without errors.
mkdir -p ~/.config/git
printf '%s %s\n' "<your-verified-github-email>" \
  "$(cat ~/.ssh/id_ed25519.pub)" > ~/.config/git/allowed_signers
git config --global gpg.ssh.allowedSignersFile ~/.config/git/allowed_signers
chmod 600 ~/.config/git/allowed_signers
```

## Why SSH and not GPG

GitHub verifies a commit signature by looking up the committer email
in its user database and then checking the signature against any
signing keys registered for that account. With GPG, the maintainer
would need to upload the GPG public key to GitHub; with SSH, the same
`id_ed25519` key already uploaded for SSH access doubles as the
signing key.
