
.PHONY: monitor

monitor:
	. ${SCRIPT_PATH}/interactive_controls

update-crontab:
	crontab < ./sysproduction/linux/crontab
