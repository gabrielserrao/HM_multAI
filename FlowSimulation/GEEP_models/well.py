well_list = ['NA1A', 'NA2', 'NA3D', 'RJS19', 'PROD005', 'PROD008', 'PROD009', 'PROD010', 'PROD012', 'PROD014', 'PROD021', 'PROD023A', 'PROD024A', 'PROD025A', 'INJ003', 'INJ005', 'INJ006', 'INJ007', 'INJ010', 'INJ015', 'INJ017', 'INJ019', 'INJ021', 'INJ022', 'INJ023', ]
well_type = ['PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'PRD', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', 'INJ', ]
well_x = [38, 21, 44, 31, 33, 19, 15, 36, 46, 50, 27, 65, 61, 57, 49, 31, 48, 59, 55, 36, 33, 29, 24, 48, 42, ]
well_y = [36, 36, 43, 27, 18, 30, 40, 42, 23, 18, 41, 23, 35, 23, 23, 19, 34, 17, 30, 28, 39, 41, 28, 11, 18, ]
from paraview.simple import *
DX = 100;
DY = 100;
X0 = 350858.5624
Y0 = 7513812.6952
well_cyl = Cylinder();
SetProperties(well_cyl,Height=1000,Radius=30);
for idx, val in enumerate(well_list):
	# wellbore
	t = Transform(well_cyl);
	t.Transform.Translate=[X0 + (well_x[idx]-0.5)*DX, Y0 + (well_y[idx]-0.5)*DY,3100];
	t.Transform.Rotate = [90,0,0];
	dp = GetDisplayProperties(t);
	if (well_type[idx] == 'PRD'):
		dp.DiffuseColor=[1,0,0];
	else:	
		dp.DiffuseColor=[0,0,1];
	Show(t);
	# well name
	title = a3DText();
	name = well_list[idx];#well_type[idx] + ': ' + well_list[idx];
	SetProperties(title,Text=name);
	tt = Transform(title);
	tt.Transform.Translate=[X0 + (well_x[idx]-0.5)*DX, Y0 + (well_y[idx]-0.5)*DY,3500];
	tt.Transform.Scale = [80, 80, 80];
	tt.Transform.Rotate = [60, 30, 0];
	dp = GetDisplayProperties(tt);
	dp.DiffuseColor = [0, 1, 0];
	Show(tt);


Render();

	
	





