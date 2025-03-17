#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "rclcpp/qos.hpp"

class FTCompensationNode : public rclcpp::Node
{
  using WrenchStamped = geometry_msgs::msg::WrenchStamped;

public:
  FTCompensationNode() : Node("ft_compensation_node")
  {
    publisher_ =
        this->create_publisher<WrenchStamped>("wrench_compensated", rclcpp::SystemDefaultsQoS());

    auto ft_callback = [this](const WrenchStamped::SharedPtr msg) -> void {
      auto ft_msg_compensated = WrenchStamped(*msg);
      publisher_->publish(ft_msg_compensated);
    };
    subscription_ = this->create_subscription<WrenchStamped>(
        "ft/force_torque_sensor_broadcaster/wrench", rclcpp::SensorDataQoS(), ft_callback);
  }

private:
  rclcpp::Publisher<WrenchStamped>::SharedPtr publisher_;
  rclcpp::Subscription<WrenchStamped>::SharedPtr subscription_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FTCompensationNode>());
  rclcpp::shutdown();
  return 0;
}