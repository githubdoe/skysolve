# skysolve
Use RaspberryPi and plate solving to take images of the night sky and identify the location of the image.
 It uses a Raspberry PI with the RPI High Quality camera and can send the solved position of were the camera is looking to a computer running SkySafari.  When mounted to a telescope and aligned to where the scope is pointing it can then be used to guide the manual pushing of the telesopce to the desired target without using any encoders on the telescope.  It communicates with SkySafari over WIFI so that no hard wired connections are needed to the computer running SKySafari.  It continually takes images and solves them about every 10 to 15 seconds so that Skysafari can always show where the scope is pointing.


There is a setup script meant to automate the many setup steps involved with configuring a Raspberry 4 running Raspbian,
so that it can run the image capture and plate solving astro application I wrote called skysolve. 

When you are ready, you can follow these steps:


1.  You will need to flash that raspbian img file to an SD card.  The easiest way to do this is to use the RaspberryPi imager
      from https://www.raspberrypi.org/software/   Download this software onto a computer that can write SD cards.  I use a Windows laptop.

2. Set the advance menu options in Raspberry pi imager.
      The Raspberry Pi Imager v1.6 has an advanced menu which is hidden away from general users just looking to write an operating system for the Pi. To activate the menu we need to press CTRL + SHIFT + X and we then gain access to advanced options that enable advanced users to customize the OS to meet their needs before they write the software to a micro SD card. 

     You need to use those advanced options to setup network parameters so that when you boot the pi it will show up on your network and you can talk to it with another computer instead of using an external display and keyboard.

     To learn more about the acvanced options see here: https://www.raspberrypi.org/blog/raspberry-pi-imager-update-to-v1-6/

     1. Set Hostname you want for the PI.   I used SKYPI.
     2. Enable SSH and set password
     3. Enable "configure WIFI" and Set SSID and password of your local WIFI router you want the pi to connect to during setup.
     4. Setup local.  This is important because it is used by the WIFI to select the right frequencies.
     

3.  Burn the image to the SD using the Raspberry Pi imager.

  
4.  Insert the SD Card into the PI after it boots in a minute or so you need to find it on your WIFI network. Your router should have assigned it an
     IP address.  Here are some ways of finding it.  https://www.raspberrypi.org/documentation/remote-access/ip-address.md

5. Connecto your computer to the PI. https://www.raspberrypi.org/documentation/remote-access/README.md
  I use VNC viewer from RealVNC on a windows PC.  So download RealVNC to your PC.  Follow the instructions for connecting to a VNC server like the RPI.



6.  Make a direcotry for skysolve then download the skysolve app from GIT Repo to your RPI and Open a Terminal Window.  You could type the following commands into Terminal to accomplish this goal.

		sudo mkdir skysolve
        cd skysolve
        sudo wget https://github.com/githubdoe/skysolve/archive/main.tar.gz
        sudo tar -xzvf main.tar.gz --strip-components=1
	
	
7.  Run the setup script 
    sudo ./skysolveSetup
	
    Here is a list of what the script does (If you want to disable or modify any of these, please edit before running the script):



- Uninstalls unattended-upgrades since they can slow down imaging and cause issues with updates.

- Updates/Upgrades the RPI

- Sets the user account to auto-login

- Sets the HDMI to force-hotplug so that it doesn't turn off HDMI when it doesn't immediately detect a display (Raspberry Pi scripts only--Helps to solve issues)

- Sets the HDMI group and HDMI mode for a certain display resolution (Raspberry Pi scripts only--Please see the comments in the scripts for details and the file /boot/config.txt on the SD card for options.)

- Disables the screen lock and screen saver to prevent them from causing issues when connecting via VNC

- Disables the CUPS modules for parallel printers since none are attached and it slows down boot (Raspberry Pi script only)








- Makes a folder called utilties on the Desktop

- Creates a shortcut/launcher for the UDev Script in utilities on the Desktop


- Creates a hotspot Wifi profile for the observing field


- Installs Astrometry.net and the Index Files

- Sets up samba and caja filesharing so you can share any folder by right clicking



mate-preferences-desktop-display.svg (Hicolor icon theme, GPL)
plip.png (Hicolor icon theme, GPL)
preferences-system-network.svg (MATE icon theme, GPL)
system-software-update.svg (Humanity icon theme, GPL)
system-upgrade.svg (MATE icon theme, GPL)

