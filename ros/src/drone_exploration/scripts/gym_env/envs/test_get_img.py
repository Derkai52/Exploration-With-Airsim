# import airsim


# drone_name = 'Drone_1'
# drone_client = airsim.MultirotorClient(ip = "127.0.0.1")
# drone_client.confirmConnection()
# drone_client.enableApiControl(True, drone_name)
# drone_client.armDisarm(True, drone_name)
# responses = drone_client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.Scene, False, False)],vehicle_name = drone_name)

import airsim #pip install airsim

# for car use CarClient() 
client = airsim.MultirotorClient()

png_image = client.simGetImage("front_left_custom", airsim.ImageType.Scene, "Drone_1")
# do something with image

