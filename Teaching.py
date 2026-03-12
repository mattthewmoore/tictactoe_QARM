from arm_control import QArmTicTacToe


def format_phi(phi):
    return f"[{phi[0]: .4f}, {phi[1]: .4f}, {phi[2]: .4f}, {phi[3]: .4f}]"


def main():
    bot = QArmTicTacToe()
    captured = []

    print("\nTeach-mode phi logger")
    print("Commands:")
    print("  [label] : capture phi with optional label (example: A1, H3)")
    print("  teach   : relax joints (PWM mode)")
    print("  hold    : re-enable position hold at current pose")
    print("  show    : print captured values")
    print("  quit    : exit\n")

    try:
        bot.enable_teach_mode()
        print("Teach mode active. Manually move the arm, then type a label and press Enter.")

        while True:
            cmd = input("cmd> ").strip()
            key = cmd.lower()

            if key in ("q", "quit", "exit"):
                break

            if key == "teach":
                bot.enable_teach_mode()
                print("Teach mode active (servos relaxed).")
                continue

            if key in ("hold", "on", "lock"):
                bot.disable_teach_mode()
                print("Position hold active.")
                continue

            if key in ("show", "list"):
                if not captured:
                    print("No captures yet.")
                else:
                    print("\nCaptured phi values:")
                    for label, phi in captured:
                        print(f"  {label}: {format_phi(phi)}")
                    print("")
                continue

            label = cmd if cmd else f"P{len(captured) + 1}"
            phi = bot.snapshot_phi()
            captured.append((label, phi))
            print(f"Saved {label}: {format_phi(phi)}")

    finally:
        # Try to leave the arm in position mode before shut down.
        try:
            bot.disable_teach_mode()
        except Exception:
            pass

        print("\nLOCATIONS_PHI = {")
        for label, phi in captured:
            print(
                f"    '{label}': [{phi[0]:.6f}, {phi[1]:.6f}, {phi[2]:.6f}, {phi[3]:.6f}],"
            )
        print("}")

        bot.terminate()


if __name__ == "__main__":
    main()