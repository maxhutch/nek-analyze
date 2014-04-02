#!/usr/bin/python
import sys
sys.path.append("/opt/visit2_7_1.linux-x86_64/2.7.1/linux-x86_64/lib/site-packages/")
import visit 
visit.LaunchNowin()

visit.OpenDatabase('./visit.nek3d')

visit.AddPlot("Pseudocolor", "temperature")
visit.SetActivePlots(0)
p = visit.PseudocolorAttributes()
p.minFlag = 1; p.maxFlag = 1
p.min = -0.0005; p.max =  0.0005
visit.SetPlotOptions(p)

visit.AddPlot("Contour", "temperature")
p = visit.ContourAttributes()
p.minFlag = 1; p.maxFlag = 1
p.min = 0; p.max =  0
p.contourNLevels = 1
p.colorType = p.ColorBySingleColor
p.singleColor = (0,0,0,255)
visit.SetActivePlots(1)
visit.SetPlotOptions(p)

visit.AddPlot("Contour", "temperature")
p = visit.ContourAttributes()
p.minFlag = 1; p.maxFlag = 1
p.min = 0.0005; p.max =  0.0005
p.contourNLevels = 1
p.colorType = p.ColorBySingleColor
p.singleColor = (0,0,255,255)
visit.SetActivePlots(2)
visit.SetPlotOptions(p)

visit.AddPlot("Contour", "temperature")
p = visit.ContourAttributes()
p.minFlag = 1; p.maxFlag = 1
p.min = -0.0005; p.max =  -0.0005
p.contourNLevels = 1
p.colorType = p.ColorBySingleColor
p.singleColor = (255,0,0,255)
visit.SetActivePlots(3)
visit.SetPlotOptions(p)

visit.SetActivePlots((0,1,2,3))
visit.AddOperator("Slice")
p = visit.SliceAttributes()
p.originIntercept = 0.000512
visit.SetOperatorOptions(p)
visit.DrawPlots()

s = visit.SaveWindowAttributes()
s.format = s.BMP
s.fileName = 'y-slice'
s.width = 1024; s.height = 768
s.screenCapture = 0
visit.SetSaveWindowAttributes(s)
visit.SaveWindow()

nimage = int(sys.argv[1])
for state in range(nimage):
  visit.SetTimeSliderState(state)
  n = visit.SaveWindow()
