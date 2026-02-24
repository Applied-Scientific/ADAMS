"""
Secure API Key Manager for ADAMS

Provides cross-platform secure storage for API keys using system keychain/credential stores:
- macOS: Keychain Access
- Linux: Secret Service API (GNOME Keyring/KWallet) or encrypted file fallback
- Windows: Windows Credential Locker (DPAPI)

Environment variables take precedence over keychain storage for Docker/CI compatibility.
"""

import logging
import os
from getpass import getpass
from pathlib import Path
from typing import Optional

try:
    import keyring
    from keyring.errors import KeyringError
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    KeyringError = Exception  # Fallback type

logger = logging.getLogger(__name__)

# Keychain constants
SERVICE_NAME = "adams-docking-agent"

# Map providers to their environment variable names and display names
PROVIDER_CONFIG = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "display_name": "OpenAI"
    },
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "display_name": "Anthropic"
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "display_name": "Google Gemini"
    }
}

def get_api_key(provider: str = "openai") -> Optional[str]:
    """
    Retrieves the API key for the specified provider with the following precedence:
    1. Environment variable (highest priority)
    2. System keychain (secure storage)
    3. Prompt user for manual entry

    Args:
        provider: The LLM provider ("openai", "anthropic", "gemini")

    Returns:
        str: The API key, or None if user cancelled.
    """
    if provider not in PROVIDER_CONFIG:
        logger.warning(f"Unknown provider: {provider}, defaulting to generic handling")
        env_var_name = f"{provider.upper()}_API_KEY"
        display_name = provider.capitalize()
    else:
        config = PROVIDER_CONFIG[provider]
        env_var_name = config["env_var"]
        display_name = config["display_name"]

    # Priority 1: Check environment variable first (Docker/CI compatibility)
    api_key = os.environ.get(env_var_name)
    if api_key:
        logger.debug(f"{display_name} API key in ENV")
        return api_key

    # Keychain username convention: provider-api-key
    keychain_username = f"{provider}-api-key"

    # Priority 2: Try to retrieve from keychain
    if KEYRING_AVAILABLE:
        try:
            api_key = keyring.get_password(SERVICE_NAME, keychain_username)
            if api_key:
                logger.debug(f"{display_name} API key in Keychain")
                os.environ[env_var_name] = api_key
                return api_key
        except KeyringError as e:
            logger.warning(f"Keychain access error: {e}")
        except Exception as e:
            logger.warning(f"Keychain error: {e}")

    # Priority 3: Prompt user for API key
    print(f"{display_name} API key not found.")
    try:
        api_key = getpass(f"Please enter your {display_name} API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return None

    # Validate that key is not empty
    if not api_key:
        print("Error: API key cannot be empty.")
        return None

    os.environ[env_var_name] = api_key

    # Offer to save to keychain
    if KEYRING_AVAILABLE:
        try:
            save_choice = input(
                f"\nDo you want to securely store this {display_name} key in your system keychain? (y/n): "
            )
        except (EOFError, KeyboardInterrupt):
            print("\nKey not stored.")
            return api_key

        if save_choice.lower() == "y":
            if save_api_key_to_keychain(api_key, provider):
                print(f"{display_name} API key securely stored in system keychain")
            else:
                print("Could not store key in keychain. It will be required for the next session.")
        else:
            print("Key not stored. It will be required for the next session.")

    return api_key


def save_api_key_to_keychain(api_key: str, provider: str = "openai") -> bool:
    """
    Securely saves the API key to the system keychain.

    Args:
        api_key: The API key to store
        provider: The provider identifier (e.g. "openai", "anthropic")

    Returns:
        bool: True if successful, False otherwise
    """
    if not KEYRING_AVAILABLE:
        logger.warning("Keyring missing")
        return False

    keychain_username = f"{provider}-api-key"

    try:
        keyring.set_password(SERVICE_NAME, keychain_username, api_key)
        logger.info(f"{provider} key stored in Keychain")
        return True
    except KeyringError as e:
        logger.error(f"Save error: {e}")
        return False
    except Exception as e:
        logger.error(f"Save error: {e}")
        return False


def delete_api_key_from_keychain(provider: str = "openai") -> bool:
    """
    Removes the API key from the system keychain.

    Args:
        provider: The provider identifier

    Returns:
        bool: True if successful or key didn't exist, False on error
    """
    if not KEYRING_AVAILABLE:
        logger.warning("Keyring missing")
        return False

    keychain_username = f"{provider}-api-key"

    try:
        keyring.delete_password(SERVICE_NAME, keychain_username)
        logger.info(f"{provider} key removed from Keychain")
        return True
    except keyring.errors.PasswordDeleteError:
        # Key didn't exist, which is fine
        logger.debug("No key in keychain to delete")
        return True
    except KeyringError as e:
        logger.error(f"Delete error: {e}")
        return False
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return False
