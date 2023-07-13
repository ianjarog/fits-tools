import numpy

def beamsize(hdr_base, hdr_top, posu, posv):

     cdelta = hdr_base["CDELT1"]
     cdeltb = hdr_base["CDELT2"]

     a = hdr_top["BMIN"]/cdelta     #radius on the x-axis 60 arcmin
     b = hdr_top["BMAJ"]/cdeltb   #radius on the y-axis 30 arcmin

     a = a/2
     b = b/2
     t = numpy.linspace(0, 2*numpy.pi, 100)
     posangle = hdr_top["BPA"]
     t_rot = numpy.deg2rad(posangle)
     Ell = numpy.array([a*numpy.cos(t) , b*numpy.sin(t)])
          #u,v removed to keep the same center location
     R_rot = numpy.array([[numpy.cos(t_rot) , -numpy.sin(t_rot)],[numpy.sin(t_rot) , numpy.cos(t_rot)]])
          #2-D rotation matrix

     Ell_rot = numpy.zeros((2,Ell.shape[1]))
     for i in range(Ell.shape[1]):
         Ell_rot[:,i] = numpy.dot(R_rot, Ell[:,i])

     u, v = (posu, posv)

     polypointsra_pix = u + Ell_rot[0,:] 
     polypointsdec_pix = v + Ell_rot[1,:]
     return {"polypointsra_pix": polypointsra_pix, "polypointsdec_pix": polypointsdec_pix}
