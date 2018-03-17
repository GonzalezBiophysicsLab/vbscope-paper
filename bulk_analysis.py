import sys
sys.path.insert(0,'/home/colin/programs/python_scripts/vbscope')
from src.ui import launch_vbscope

fdir = '/media/colin/Seagate Backup Plus Drive/colin/microscope/20180314 vc436/'

align_filename = 'vbscope_alignment_888_2x_20180316.dat'

import os
fnames = []
for root, dirs, files in os.walk(fdir):
	for filename in files:
		if filename.endswith('.tif'):
			if filename.startswith('lx') or filename.startswith('hx'):
				fnames.append([root,filename])


gui = launch_vbscope(True)
gui.closeEvent = gui.unsafe_close

gui.docks['transform'][1].load(align_filename)


for i in range(len(fnames)):
	root,filename = fnames[i]
	fname = os.path.join(root,filename)
	gui.load(fname)

	gui.docks['background'][1].combo_method.setCurrentIndex(4)
	gui.docks['background'][1].update_background()

	gui.docks['spotfind'][1].sliders[0].setValue(1000) # p=1.0
	gui.docks['spotfind'][1].sliders[0].setValue(900) # p = 0.90
	gui.docks['spotfind'][1].spin_start.setValue(9)
	gui.docks['spotfind'][1].spin_end.setValue(200)
	gui.docks['spotfind'][1].findspots()

	# gui.docks['transform'][1].stochastic()

	gui.docks['extract'][1].extract()
	gui.docks['extract'][1].launch_plotter()
	gui.docks['extract'][1].ui_p.closeEvent = gui.docks['extract'][1].ui_p.unsafe_close
	# gui.docks['extract'][1].ui_p.data.remove_dead(10.)
	print os.path.join(root,'traces_%s.dat'%(filename))
	gui.docks['extract'][1].ui_p.export_traces(os.path.join(root,'traces_%s.dat'%(filename)))
	gui.docks['extract'][1].ui_p.close()

gui.close()


def send_finished_email():
	import smtplib
	from email.mime.text import MIMEText

	msg = MIMEText("Finished job on desktop computer")

	me = 'ckinztho@gmail.com'
	you = 'ckinztho@gmail.com'

	msg['Subject'] = 'Desktop job finished'
	msg['From'] = me
	msg['To'] = you

	s = smtplib.SMTP('smtp.gmail.com',587)
	s.ehlo()
	s.starttls()
	s.login('ckinztho','tuowqwzbngehknhc')
	s.sendmail(me, [you], msg.as_string())
	s.quit()

send_finished_email()
