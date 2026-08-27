# Third-party notices

## LTX-2

This project is based on LTX-2 by Lightricks Ltd. Original copyright, license,
patent, trademark, and attribution notices are retained. See `LICENSE` for the
terms applicable to this repository.

## MSST-WebUI

Echo 1.5 voice-memory inference depends on
[MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI), pinned by the setup
script to commit `43e30b860c611b516ed9b67c75a56792a67ec902`.

MSST-WebUI is a separate work licensed under the GNU Affero General Public
License v3.0. Its source checkout, license, and Git history live under
`third_party/MSST-WebUI` after setup and are not vendored into this repository.

The Bandit checkpoint and configuration are distributed by the MSST projects
and remain subject to their upstream terms:

- Model: `model_bandit_plus_dnr_sdr_11.47.chpt`
- Model source: <https://huggingface.co/Sucial/MSST-WebUI>
- Original model project: <https://github.com/ZFTurbo/Music-Source-Separation-Training>
