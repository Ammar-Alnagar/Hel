/*
 * ═══════════════════════════════════════════════════════════════════════════
 * ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 * ░░                                                                       ░░
 * ░░                  ⚠️ ☠️  HERE BE DRAGONS  ☠️ ⚠️                        ░░
 * ░░                                                                       ░░
 * ░░   ⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸⸸   ░░
 * ░░                                                                       ░░
 * ░░            🔥 ABANDON ALL HOPE, YE WHO ENTER HERE 🔥                 ░░
 * ░░                                                                       ░░
 * ░░   ┌─────────────────────────────────────────────────────────────┐   ░░
 * ░░   │                                                             │   ░░
 * ░░   │  Greetings, foolish mortal who stumbles into this abyss.   │   ░░
 * ░░   │                                                             │   ░░
 * ░░   │  What lies before you is not code—it is a CURSE.           │   ░░
 * ░░   │  A labyrinth of logic so twisted, so arcane, that it       │   ░░
 * ░░   │  defies comprehension itself.                               │   ░░
 * ░░   │                                                             │   ░░
 * ░░   │  This monstrosity is held together by:                      │   ░░
 * ░░   │    • Duct tape (metaphorical and spiritual)                │   ░░
 * ░░   │    • Prayers whispered at 3 AM                             │   ░░
 * ░░   │    • Stack Overflow answers from 2009                      │   ░░
 * ░░   │    • Pure, unfiltered desperation                          │   ░░
 * ░░   │    • The tears of junior developers                        │   ░░
 * ░░   │                                                             │   ░░
 * ░░   │  Once, two beings understood this code:                     │   ░░
 * ░░   │             ⚡ God and Me ⚡                                 │   ░░
 * ░░   │                                                             │   ░░
 * ░░   │  Now... I have forgotten.                                   │   ░░
 * ░░   │  Only God remains.                                          │   ░░
 * ░░   │  And I'm not sure He's still watching.                      │   ░░
 * ░░   │                                                             │   ░░
 * ░░   └─────────────────────────────────────────────────────────────┘   ░░
 * ░░                                                                       ░░
 * ░░               ⚔️  YOUR OPTIONS, BRAVE WARRIOR:  ⚔️                   ░░
 * ░░                                                                       ░░
 * ░░         1. Turn back now. Close this file. Walk away.                ░░
 * ░░         2. Proceed at your own peril and join the fallen.            ░░
 * ░░         3. Rewrite from scratch (recommended).                       ░░
 * ░░                                                                       ░░
 * ░░   If you choose Option 2, know this:                                 ░░
 * ░░   • Touching ANYTHING may cause catastrophic failure                 ░░
 * ░░   • The tests pass by ACCIDENT, not design                           ░░
 * ░░   • Production stability is a MYTH we tell ourselves                 ░░
 * ░░   • Your IDE's warnings are trying to SAVE you                       ░░
 * ░░                                                                       ░░
 * ░░                      ⚰️  MEMORIAL WALL  ⚰️                           ░░
 * ░░              For those who tried and failed before you:              ░░
 * ░░                                                                       ░░
 * ░░                    • Junior Dev (2019-2019) RIP                      ░░
 * ░░                    • Senior Dev (2020-2020) RIP                      ░░
 * ░░                    • Tech Lead (2021-2022) Missing                   ░░
 * ░░                    • That One Intern (2023-2023) Therapy            ░░
 * ░░                                                                       ░░
 * ░░   ═══════════════════════════════════════════════════════════════   ░░
 * ░░                                                                       ░░
 * ░░                         May God have mercy.                          ░░
 * ░░                         Godspeed, you magnificent fool.              ░░
 * ░░                         You're going to need it.                     ░░
 * ░░                                                                       ░░
 * ░░                    Last Modified: DO NOT MODIFY                      ░░
 * ░░                    Last Successful Edit: NEVER                       ░░
 * ░░                    Next Refactor: HEAT DEATH OF UNIVERSE             ░░
 * ░░                                                                       ░░
 * ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 * ═══════════════════════════════════════════════════════════════════════════
 */


#include "app.hpp"
#include <iostream>
#include <string>
#include <vector>

InferenceArgs parse_args(int argc, char* argv[]) {
    InferenceArgs args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            args.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            args.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            args.top_p = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoi(argv[++i]);
        } else if (arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "--help") {
            App::print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            App::print_usage(argv[0]);
            exit(1);
        }
    }

    // Validate required arguments
    if (args.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        App::print_usage(argv[0]);
        exit(1);
    }

    if (args.prompt.empty()) {
        std::cerr << "Error: --prompt is required" << std::endl;
        App::print_usage(argv[0]);
        exit(1);
    }

    return args;
}

int main(int argc, char* argv[]) {
    InferenceArgs args = parse_args(argc, argv);

    std::cout << "Helios Engine - Mini LLM Inference" << std::endl;
    std::cout << "===================================" << std::endl;

    return App::run(args);
}
