
nnode = 300000
ntime = 24*30
nelem = 100000
nspectra = 1
nfreq = 1
ndir = 1


vars =[
  dict(name="u" ,type="f4" ,units="m/s" ,standard_name="" ,long_name=""),
  dict(name="v" ,type="f4" ,units="m/s" ,standard_name="" ,long_name=""),
  dict(name="w" ,type="f4" ,units="m/s" ,standard_name="" ,long_name=""),
]

spectravar = [
  dict(name="e" ,type="f4" ,units="watts" ,standard_name="" ,long_name=""),
]


intput1 = dict(
  name="test",
  folder=None,
  metadata=dict(
    title="Mytitle"
  ),
  nc=dict(
    dimensions = [
      dict(name="npe" ,value=3),
      dict(name="nnode" ,value=nnode),
      dict(name="ntime" ,value=nelem),
      dict(name="nelem" ,value=nelem),
    ],
    variables=[
      dict(name="b" ,type="f4" ,shape=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="lat" ,type="f8" ,shape=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="lng" ,type="f8" ,shape=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="elem" ,type="i4" ,shape=["nelem"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="time" ,type="f8" ,shape=["ntime"] ,units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
    ]),
  nca = dict(
    dimensions = [
      dict(name="ntime" ,value=ntime),
      dict(name="nnode" ,value=nnode),
      dict(name="nspectra" ,value=nspectra),
      dict(name="nfreq" ,value=nfreq),
      dict(name="ndir" ,value=ndir),
    ],
    groups=[
      dict(name="s" ,strshape=["ntime", "nnode"] ,variables=vars),
      dict(name="t" ,strshape=["nnode" ,"ntime"] ,variables=vars),
      dict(name="ss" ,strshape=["ntime", "nspectra", "nfreq", "ndir"] ,variables=spectravar),
      dict(name="st" ,strshape=["nspectra" ,"ntime", "nfreq", "ndir"] ,variables=spectravar)
    ]
  )
)



