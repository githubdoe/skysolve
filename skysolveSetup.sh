#!/bin/bash
#
#   Installs files needed for skySolve program  Script based in part on
#	AstroRaspbianPi Raspberry Pi Raspbian KStars/INDI Configuration Script
#ï»¿  Copyright (C) 2018 Robert Lancaster <rlancaste@gmail.com>
#	This script is free software; you can redistribute it and/or
#	modify it under the terms of the GNU General Public
#	License as published by the Free Software Foundation; either
#	version 2 of the License, or (at your option) any later version.
#Text Format


if [ "$(whoami)" != "root" ]; then
	echo "Please run this script with sudo due to the fact that it must do a number of sudo tasks.  Exiting now."
	exit 1
elif [ -z "$BASH_VERSION" ]; then
	echo "Please run this script in a BASH shell because it is a script written using BASH commands.  Exiting now."
	exit 1
else
	echo "You are running BASH $BASH_VERSION as the root user."
fi

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
YEL='\e[38;2;255;255;0m'
GRE='\e[38;0;255;0;0m'
DEF='\e[m'
BOL='\e[1m'


function display
{
    echo ""
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~ $*"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    
    # This will display the message in the title bar (Note that the PS1 variable needs to be changed too--see below)
    echo -en "\033]0;SkySolve-$*\a"
}


function checkForConnection
{
		testCommand=$(curl -Is $2 | head -n 1)
		echo "testCommand results $testCommand"
		if [[ "${testCommand}" == *"200"* || "${testCommand}" == *"301"* ]]
  		then 
  			echo "$1 was found. The script can proceed."
  		else
  			echo "$1, ($2), a required connection, was not found, aborting script."
  			echo "If you would like the script to run anyway, please comment out the line that tests this connection in this script."
  			exit
		fi
}



display "This will update, install and configure your Raspberry Pi to work with SkySolve. Be sure to read the script first to see what it does and to customize it."
echo -e $YEL"Welcome to the SKySolve Configuration Script."$DEF
echo It can update the operating system, install the Field hot spot, and the plate solve code.
read -p "Are you ready to proceed (y/n)? " proceed

if [ "$proceed" != "y" ]
then
	exit
fi


export USERHOME=$(sudo -u $SUDO_USER -H bash -c 'echo $HOME')


# This changes the UserPrompt for the Setup Script (Necessary to make the messages display in the title bar)
PS1='SkySolve Setup~$ '

read -p "Do you want to update the PI operating system (Recommended) (y/n)? " proceed

if [ "$proceed" == "y" ]
then

    #########################################################
    #############  Updates

    # This would update the Raspberry Pi kernel.  For now it is disabled because there is debate about whether to do it or not.  To enable it, take away the # sign.
    #display "Updating Kernel"
    #sudo rpi-update 

    # Updates the Raspberry Pi to the latest packages.
    display "Updating installed packages"
    sudo apt update
    sudo apt -y upgrade
    sudo apt -y dist-upgrade

    #update GPIO module
    wget https://project-downloads.drogon.net/wiringpi-latest.deb
    sudo dpkg -i wiringpi-latest.deb
fi

#########################################################
#############  Configuration for Ease of Use/Access

# This makes sure there is a config folder owned by the user, since many things depend on it.
mkdir -p $USERHOME/.config
sudo chown $SUDO_USER:$SUDO_USER $USERHOME/.config

# This will set up the Pi so that double clicking on desktop icons brings up the program right away
# The default behavior is to ask what you want to do with the executable file.
display "Setting desktop icons to open programs when you click them."
if [ -f $USERHOME/.config/pcmanfm-qt/lxqt/settings.conf ]
then
	sed -i "s/QuickExec=false/QuickExec=true/g" $USERHOME/.config/pcmanfm-qt/lxqt/settings.conf
fi
if [ -f $USERHOME/.config/pcmanfm-qt/default/settings.conf ]
then
	sed -i "s/QuickExec=false/QuickExec=true/g" $USERHOME/.config/pcmanfm-qt/default/settings.conf
fi
if [ -f $USERHOME/.config/libfm/libfm.conf ]
then
	if [ -z "$(grep 'quick_exec' $USERHOME/.config/libfm/libfm.conf)" ]
	then
		sed -i "/\[config\]/ a quick_exec=1" $USERHOME/.config/libfm/libfm.conf
	else
		sed -i "s/quick_exec=0/quick_exec=1/g" $USERHOME/.config/libfm/libfm.conf
	fi
fi
if [ -f /etc/xdg/libfm/libfm.conf ]
then
	if [ -z "$(grep 'quick_exec' /etc/xdg/libfm/libfm.conf)" ]
	then
		sed -i "/\[config\]/ a quick_exec=1" /etc/xdg/libfm/libfm.conf
	else
		sed -i "s/quick_exec=0/quick_exec=1/g" /etc/xdg/libfm/libfm.conf
	fi
fi

# This will set your account to autologin.  If you don't want this. then put a # on each line to comment it out.
display "Setting account: "$SUDO_USER" to auto login."
if [ -n "$(grep '#autologin-user' '/etc/lightdm/lightdm.conf')" ]
then
	sed -i "s/#autologin-user=/autologin-user=$SUDO_USER/g" /etc/lightdm/lightdm.conf
	sed -i "s/#autologin-user-timeout=0/autologin-user-timeout=0/g" /etc/lightdm/lightdm.conf
fi



# This pretends an HDMI display is connected at all times, otherwise, the pi might shut off HDMI
# So that when you go to plug in an HDMI connector to diagnose a problem, it doesn't work
# This makes the HDMI output always available
if [ -n "$(grep '#hdmi_force_hotplug=1' '/boot/config.txt')" ]
then
	sed -i "s/#hdmi_force_hotplug=1/hdmi_force_hotplug=1/g" /boot/config.txt
fi

# This sets the group for the HDMI mode.  Please see the config file for details about all the different modes
# There are many options.  I selected group 1 mode 46 because that matches my laptop's resolution.
# You might want a different mode and group if you want a certain resolution in VNC
if [ -n "$(grep '#hdmi_group=1' '/boot/config.txt')" ]
then
	sed -i "s/#hdmi_group=1/hdmi_group=2/g" /boot/config.txt
fi

# This sets the HDMI mode.  Please see the config file for details about all the different modes
# There are many options.  I selected group 1 mode 46 because that matches my laptop's resolution.
# You might want a different mode and group if you want a certain resolution in VNC
if [ -n "$(grep '#hdmi_mode=1' '/boot/config.txt')" ]
then
	sed -i "s/#hdmi_mode=1/hdmi_mode=46/g" /boot/config.txt
fi

# This comments out a line in Raspbian's config file that seems to prevent the desired screen resolution in VNC
# The logic here is that if the line does exist, and if the line is not commented out, comment it out.
if [ -n "$(grep '^dtoverlay=vc4-kms-v3d' '/boot/config.txt')" ]
then
	sed -i "s/dtoverlay=vc4-kms-v3d/#dtoverlay=vc4-kms-v3d/g" /boot/config.txt
fi
if [ -n "$(grep '^dtoverlay=vc4-fkms-v3d' '/boot/config.txt')" ]
then
	sed -i "s/dtoverlay=vc4-fkms-v3d/#dtoverlay=vc4-fkms-v3d/g" /boot/config.txt
fi


# This will prevent the raspberry pi from turning on the lock-screen / screensaver which can be problematic when using VNC
if [ -z "$(grep 'xserver-command=X -s 0 dpms' '/etc/lightdm/lightdm.conf')" ]
then
	sed -i "/\[Seat:\*\]/ a xserver-command=X -s 0 dpms" /etc/lightdm/lightdm.conf
fi

display "Setting up hot spot"

if [ ! -f AccessPopup.tar.gz ] 
then
    display "setup networking hotpot"
    curl "https://www.raspberryconnect.com/images/scripts/AccessPopup.tar.gz" -o AccessPopup.tar.gz
    tar -xvf ./AccessPopup.tar.gz

fi
# allow change of hot spot ap ssid name and password change
ap_ssid_change()
{   script_path="/usr/bin/"
    scriptname="accesspopup"
	if [ -f "$script_path$scriptname" ] >/dev/null 2>&1; then 
		echo -e "The current ssid and password for the Field hot spot access point are:"
		ss="$( grep -F 'ap_ssid=' $script_path$scriptname )"
		echo "SSID:${ss:8}"
		pw="$( grep -F 'ap_pw=' $script_path$scriptname )"
		echo "Password:${pw:6}"
		prof="$( grep -F 'ap_profile_name=' $script_path$scriptname )"
		echo -e $YEL"Enter the new SSID"$DEF
		echo "Press enter to keep the existing SSID"
		read newss
		if [ ! -z "$newss" ]; then
			sed -i "s/ap_ssid=.*/ap_ssid='$newss'/" "$script_path$scriptname"
		fi
		echo -e $YEL"Enter the new Password"$DEF
		echo "The password must be at least 8 characters"
		echo "Press enter to keep the existing Pasword"
		read newpw
		if [ ! -z "Snewpw" ] && [ ${#newpw} -ge 8 ]; then
			sed -i "s/ap_pw=.*/ap_pw='$newpw'/" "$script_path$scriptname"
		fi
		echo "The Access Points SSID and Password are:"
		ss="$( grep -F 'ap_ssid=' $script_path$scriptname )"
		pw="$( grep -F 'ap_pw=' $script_path$scriptname )"
		echo "SSID:${ss:8}"
		echo "Password: ${pw:6}"
		#remove AP profile
		pro="$(nmcli -t -f NAME con show | grep ${prof:17:-1} )"
		if [ ! -z "$pro" ]; then
			nmcli con delete "$pro" >/dev/null 2>&1
			nmcli con reload
		fi
	else
		echo "$scriptname is not available."
		echo "Please install the script first"
	fi
}
# This will install the autohotspot files so that the pi can connect to local wifi or be a hotspot.
echo -e $YEL"This will install the Field hotspot wifi into the PI"$DEF
echo "It will ask you for options."
echo "The two options keys to press are 1 - install hot spot and then 9 - to exit"
echo "You can also setup you local/Home wifi by pressing option 5 before you exit with 9."
echo
echo -e $YEL"Press any key to start the setup."$DEF
read   

cd AccessPopup
sudo ./installconfig.sh
ap_ssid_change
cd ..

display "setting up pyhton libraries"
#install python scipy
echo "installing extra python modules"
sudo apt --fix-broken install
sudo apt install python3-scipy
sudo apt install python3-matplotlib
echo "This next step may take 10 to 15 minutes."
sudo apt install python3-numpy
echo "installing opencv"
sudo apt install python3-opencv
sudo pip install --break-system-packages flask-sock

# Installs the Astrometry.net package for supporting offline plate solves.  
display "Installing Astrometry.net"
sudo apt -y install astrometry.net





#########################################################

#!/bin/bash

#	AstroPi3 Astrometry Index File Installer
#ï»¿  Copyright (C) 2018 Robert Lancaster <rlancaste@gmail.com>
#	This script is free software; you can redistribute it and/or
#	modify it under the terms of the GNU General Public
#	License as published by the Free Software Foundation; either
#	version 2 of the License, or (at your option) any later version.

read -p "do you want to install astrometry index files? (do at least once)" answer
if [ "$answer" == "y" ]
    then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "Welcome to the Astrometry Index Installer Script"
    echo "This script will ask you which Index files you want to download and then will install them to /usr/share/astrometry"
    echo "Note that you need to install at least the index files that cover 10% to 100% of your FOV."
    echo "Please make sure you know your FOV before Proceeeding."
    echo "For the usual lenses (up to 25mm) on the PI quality cam the field will be 10 degs or more"	
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "To download and install the correct files, you need to decide which packages you want."
    echo "Note that for large file sizes, the index files are in a big set."
    echo "If you type the word 'large', you will get index files 4208-4219 which covers 30 arcmin to 2000 arcmin."
    echo "For smaller fields, the file sizes become much bigger, so they are in separate packages."
    echo "You just need to type the file number to download and install that package"
    echo "You can select more than one packages by typing each number after the other separated by spaces or commas"
    echo "Here is a list of all the available index file sets and their FOV's in Arc minutes"
    echo "File  FOV"
    echo "4207  22 - 30"
    echo "4206  16 - 22"
    echo "4205  11 - 16"
    echo "4204  8 - 11"
    echo "4203  5.6 - 8.0"
    echo "4202  4.0 - 5.6"
    echo "4201  2.8 - 4.0"
    echo "4200  2.0 - 2.8"
    echo "RPI25FL for RPI HiQuality Camera with 25 FL Lens"
    echo ""
    read -p "Which file set would you like to download? Remember, type either 'large' or the file number(s) above: (large recommended)" indexFile

    if [ "$indexFile" = "RPI25FL" ]
    then 
        wget http://data.astrometry.net/debian/astrometry-data-4208-4219_0.45_all.deb
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4112.fits
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4113.fits
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4114.fits
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4115.fits    
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4116.fits
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4117.fits
        wget -P /usr/share/astrometry/ http://data.astrometry.net/4100/index-4118.fits     
    fi
    if [[ $indexFile = *"large"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4208-4219_0.45_all.deb
    fi

    if [[ $indexFile = *"4207"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4207_0.45_all.deb
    fi

    if [[ $indexFile = *"4206"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4206_0.45_all.deb
    fi

    if [[ $indexFile = *"4205"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4205_0.45_all.deb
    fi

    if [[ $indexFile = *"4204"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4204_0.45_all.deb
    fi

    if [[ $indexFile = *"4203"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4203_0.45_all.deb
    fi

    if [[ $indexFile = *"4202"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4202_0.45_all.deb
    fi

    if [[ $indexFile = *"4201"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4201-1_0.45_all.deb
        wget http://data.astrometry.net/debian/astrometry-data-4201-2_0.45_all.deb
        wget http://data.astrometry.net/debian/astrometry-data-4201-3_0.45_all.deb
        wget http://data.astrometry.net/debian/astrometry-data-4201-4_0.45_all.deb
    fi

    if [[ $indexFile = *"4200"* ]]
    then
        wget http://data.astrometry.net/debian/astrometry-data-4200-1_0.45_all.deb
        wget http://data.astrometry.net/debian/astrometry-data-4200-2_0.45_all.deb
        wget http://data.astrometry.net/debian/astrometry-data-4200-3_0.45_all.deb
        wget http://data.astrometry.net/debian/astrometry-data-4200-4_0.45_all.deb
    fi

    sudo dpkg -i astrometry-data-*.deb
    sudo rm *.deb
fi

display "Enabling skysolve and Skysafari interface"

#setup auto run of encoder and skysolve at boot.

    echo "setup encoder service"
    sudo cp /home/pi/skysolve/encodertoSkySafari.service /etc/systemd/system/encodertoSkySafari.service
    sudo systemctl enable encodertoSkySafari.service

    echo "setup skySolve service"
    sudo cp /home/pi/skysolve/skysolve.service /etc/systemd/system/skysolve.service
    sudo systemctl enable skysolve.service


read -p "Is this an RPI 5 or later (y/n)? " pi5

if [ "$pi5" == "y" ]
then
    sudo apt install python3-rpi-lgpio
    apt --fix-broken install


fi

echo -e $YEL"Your requested installations are complete.You should restart your Pi by typing: sudo reboot"$DEF 
display "Script Execution Complete.  Your Raspberry should now be ready to use for SkySolve."
