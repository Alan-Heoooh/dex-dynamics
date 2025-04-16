#include <string.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

#include "xhand_control.hpp"
std::atomic<bool> exit_flag(false);

// Signal handler for SIGINT
void signalHandler(int signum) {
  std::cout << "\nInterrupt signal (" << signum << ") received.\n";
  exit_flag.store(true);
}
int main() {
  std::signal(SIGINT, signalHandler);
  xhand_control::XHandControl xhand_control;
  auto ifnames = xhand_control.enumerate_devices("EtherCAT");

  if (ifnames.empty()) {
    std::cout << "No EtherCAT devices found" << std::endl;
    return 0;
  }

  for (const auto& ifname : ifnames) {
    std::cout << ifname << std::endl;
  }

  std::string device_name{ifnames[0]};

  auto ret = xhand_control.open_ethercat(device_name);
  if (ret) {
    std::cout << "Ethercat opened successfully" << std::endl;
    auto hand_list = xhand_control.list_hands_id();
    for (auto hand_id : hand_list) {
      std::cout << "hand id: " << (int)hand_id << std::endl;
    }

    {
      auto ret = xhand_control.get_hand_type(hand_list[0]);
      if (ret.first)
        std::cout << "type: " << ret.second << std::endl;
      else
        std::cout << "get type failed:" << ret.first.error_code
                  << ret.first.error_message << std::endl;
    }

    // {
    //   auto ret = xhand_control.get_hand_type(hand_list[0]);
    //   if (ret.first)
    //     std::cout << "type: " << ret.second << std::endl;
    //   else
    //     std::cout << "get type failed:" << ret.first.error_code
    //               << ret.first.error_message << std::endl;
    // }

    // {
    //   auto ret = xhand_control.set_hand_name(hand_list[0], "xhand");
    //   if (ret)
    //     std::cout << "set name successfully,current name "
    //               << xhand_control.get_hand_name(hand_list[0]).second
    //               << std::endl;
    //   else
    //     std::cout << "set name failed:" << ret.error_code;
    // }

    // {
    //   // sensor id is [0x11,0x15], only call if needed
    //   auto ret = xhand_control.reset_sensor(hand_list[0], 0x11);
    //   if (ret)
    //     std::cout << "reset sensor successfully" << std::endl;
    //   else
    //     std::cout << "reset sensor failed:" << ret.error_code << std::endl;
    // }

    {
      // read palm version
      auto ret = xhand_control.read_version(hand_list[0], 0x80 | hand_list[0]);
      if (ret.first) {
        std::cout << " plam version: " << ret.second << std::endl;
      } else {
        std::cout << "read version failed:" << ret.first.error_code << "  "
                  << ret.first.error_message << std::endl;
      }
    }

    {
      auto ret = xhand_control.read_version(hand_list[0], 0);
      if (ret.first) {
        std::cout << " finger 0 version: " << ret.second << std::endl;
      } else {
        std::cout << "read version failed:" << ret.first.error_code << "  "
                  << ret.first.error_message << std::endl;
      }
    }

    // read hand parameter
    // {
    //   auto ret = xhand_control.read_parameters(hand_list[0]);
    //   if (ret.first) {
    //     for (int i = 0; i < 12; i++) {
    //       printf("finger %d parameter: %d,%d,%d,%d,%d,%d\n", i,
    //              ret.second.position_stretching[i],
    //              ret.second.position_closing[i], ret.second.position_zero[i],
    //              ret.second.position_init[i], ret.second.angle_stretching[i],
    //              ret.second.angle_closing[i]);
    //     }
    //   } else {
    //     std::cout << "read hand parameter failed:" << ret.first.error_code
    //               << "  " << ret.first.error_message << std::endl;
    //   }
    // }

    // {
    //   std::cout << "press y or n to calibrate finger joint" << std::endl;
    //   std::string input;
    //   std::cin >> input;
    //   std::cin.get();
    //   if (input == "y") {
    //     for (int i = 1; i < 6; ++i) {
    //       // wait for user to press enter the button
    //       std::vector<int8_t> angles_limit = {
    //           0, 90,  -70, 80,  0, 90,  12, -8,  0, 110, 0, 110,
    //           0, 110, 0,   110, 0, 110, 0,  110, 0, 110, 0, 110};
    //       std::cout << "press enter to calibrate joint,step " << i <<
    //       std::endl; std::cin.get(); auto ret =
    //           xhand_control.calibrate_joint(hand_list[0], i, angles_limit);
    //       if (ret) {
    //         std::cout << "calibrate successfully" << std::endl;
    //       }
    //       if (ret.error_code != 0) {
    //         std::cout << "error code: " << ret.error_code
    //                   << "error description: " << ret.error_message
    //                   << std::endl;
    //       }
    //     }
    //   }
    // }

    {
      std::cout << "press y or n to calibrate finger joint" << std::endl;
      std::string input;
      std::cin >> input;
      std::cin.get();
      bool calibration_success = false;
      if (input == "y") {
        while (!exit_flag.load() && calibration_success == false)
        {
          // wait for user to press enter the button
          std::vector<int8_t> calibrated_angles = {
            10, 60, 30, 0, 10, 10, 10, 10, 10, 10, 10, 10
          };

          std::vector<int8_t> angles_limit = {
              0, 105, -60, 90, -10, 105, 10, -10, 0, 110, 0, 110, 
              0, 110, 0, 110, 0, 110, 0, 110, 0, 110, 0, 110};

          // MH2.25
          std::vector<int8_t> lower_angles = {
              0, -60, -10, -10, 0, 0, 0, 0, 0, 0, 0, 0
          };
          std::vector<int8_t> upper_angles = {
              105, 90, 105, 10, 110, 110, 110, 110, 110, 110, 110, 110
          };

          for (size_t i = 0; i < lower_angles.size(); i++)
          {
            angles_limit[i * 2] = lower_angles[i];
            angles_limit[i * 2 + 1] = upper_angles[i];
          }
          

          std::string tools = "S1";
          uint8_t step = 1;

          // uint8_t device_id = 0x80;
          
          auto ret = xhand_control.calibrate_joint_by_mold(hand_list[0], tools,step, calibrated_angles, angles_limit);

          if (ret) {
            std::cout << "command sent successfully" << std::endl;
            calibration_success = true;
          } else {
            std::cout << "\ncommand sent failed:\ncode: " << ret.error_code << "\nmsg: "
                      << ret.error_message << std::endl;
            std::cout << "\npress y or n to calibrate finger joint" << std::endl;
            std::string input;
            std::cin >> input;
            std::cin.get();
          }
        }
        
      }
    }

    // {
    //   std::cout << "press y or n to upgrade finger" << std::endl;
    //   std::string input;
    //   std::cin >> input;
    //   if (input == "y") {
    //     for (int i = 0; i < 12; i++) {
    //       auto ret = xhand_control.upgrade_device(
    //           hand_list[0], i,
    //           "/root/work/firmware/jointboard_g0_v107_20241015180220.bin");
    //       if (ret) {
    //         std::cout << "upgrade finger successfully" << std::endl;
    //       } else {
    //         std::cout << "upgrade finger failed:" << ret.error_code << "  "
    //                   << ret.error_message << std::endl;
    //         break;
    //       }
    //     }
    //   }
    // }
    // {
    //   // user confirm if upgrade palm y or n
    //   std::cout << "press y or n to upgrade palm" << std::endl;
    //   std::string input;
    //   std::cin >> input;
    //   if (input == "y") {
    //     auto ret = xhand_control.upgrade_device(
    //         hand_list[0], 0x80 | hand_list[0],
    //         "/root/work/firmware/commboard_v113_20241015175752.bin");
    //     if (ret) {
    //       std::cout << "upgrade palm successfully" << std::endl;
    //     } else {
    //       std::cout << "upgrade palm failed:" << ret.error_code << "  "
    //                 << ret.error_message << std::endl;
    //     }
    //   }
    // }

    // if (xhand_control.reset_hand(hand_list[0])) {
    //   std::cout << "reset hand successfully" << std::endl;
    // }
    // sensor id is [0x11,0x15], only call if needed
    // if (xhand_control.reset_sensor(hand_list[0], 0x11)) {
    //   std::cout << "reset sensor successfully" << std::endl;
    // }
  } else {
    std::cout << "Failed to open EtherCAT device" << ret.error_code
              << ret.error_message << std::endl;
    return 0;
  }
  auto hand_list = xhand_control.list_hands_id();
  // Main loop
  HandCommand_t command;
  memset(&command, 0, sizeof(command));
  for (int i = 0; i < 12; i++) {
    command.finger_command[i].id = i;
    command.finger_command[i].position = 0;
    command.finger_command[i].kp = 0;
    command.finger_command[i].kd = 0;
    command.finger_command[i].mode = 3;
  }
  while (!exit_flag.load()) {
    std::cout << "Enter finger id: ";
    int finger_id = 0;
    std::cin >> finger_id;  // Wait for user input
    if (finger_id < 0 || finger_id > 11) {
      std::cout << "Invalid finger id " << finger_id << std::endl;
      continue;
    }
    if (exit_flag.load()) break;
    std::cout << "Enter position: ";
    float position;
    std::cin >> position;  // Wait for user input
    if (exit_flag.load()) break;

    command.finger_command[finger_id].position = position;
    command.finger_command[finger_id].kp = 100;
    command.finger_command[finger_id].kd = 0;
    command.finger_command[finger_id].mode = 3;
    command.finger_command[finger_id].tor_max = 350;
    auto ret = xhand_control.send_command(hand_list[0], command);
    if (ret) {
      std::cout << "command sent successfully" << std::endl;
    } else {
      std::cout << "command sent failed" << ret.error_code << "  "
                << ret.error_message << std::endl;
    }

    {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      auto ret = xhand_control.read_state(hand_list[0]);
      if (ret.first) {
        auto state = ret.second;
        // int finger_id = 0;
        std::cout << "Finger " << finger_id
                  << " position: " << (state.finger_state[finger_id].position)
                  << " control mode: "
                  << (state.finger_state[finger_id].raw_position & 0xf)
                  << " raw position: "
                  << ((state.finger_state[finger_id].raw_position >> 4) & 0xfff)
                  << " torq: " << state.finger_state[finger_id].torque
                  << std::endl;
        // for (int finger_id = 0; finger_id < 1; finger_id++) {
        //   auto state = ret.second;
        //   std::cout << "Finger " << finger_id << " position: "
        //             << ((state.finger_state[finger_id].raw_position >> 4) &
        //                 0x0fff)
        //             << " control mode: "
        //             << (state.finger_state[finger_id].raw_position & 0xf)
        //             << " raw position: "
        //             << state.finger_state[finger_id].raw_position <<
        //             std::endl;
        // }
      } else {
        std::cout << "read state failed" << ret.first.error_code << "  "
                  << ret.first.error_message << std::endl;
      }
    }

    // std::cout << "Finger " << finger_id
    //           << " position: " << state.finger_state[finger_id].position
    //           << std::endl;
  }
  xhand_control.close_device();
  std::cout << "Exiting program.\n";
  return 0;
}