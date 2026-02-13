# Human Tracking Android App

This is the source code for the Android Client of the Human Tracking project. It connects to the PC server to display the real-time tracked video feed.

## Prerequisites

1.  **Flutter SDK**: [Install Flutter](https://docs.flutter.dev/get-started/install/windows)
2.  **Android Studio**: Required for building the APK.

## Setup

1.  Open this folder (`App`) in your terminal or VS Code.
2.  Get dependencies:
    ```bash
    flutter pub get
    ```

## Configuration

1.  Open `lib/main.dart`.
2.  Find the line:
    ```dart
    final String serverUrl = 'http://10.0.2.2:5000/video_feed';
    ```
3.  **IMPORTANT**:
    *   If running on **Android Emulator**, keep `10.0.2.2`.
    *   If running on a **Real Phone**, change this to your PC's IP address (e.g., `http://192.168.1.15:5000`).
    *   Ensure both Phone and PC are on the **same Wi-Fi network**.

## Building the APK

To generate the installable APK file:

```bash
flutter build apk --release
```

The APK will be found at:
`build/app/outputs/flutter-apk/app-release.apk`

Transfer this file to your phone and install it!
