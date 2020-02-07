
nnode = 262145
ntime = 2
nelem = 1000
nspectra = 300
nfreq = 33
ndir = 36


vars =[
  dict(name="u" ,type="f4" ,units="m/s" ,standard_name="" ,long_name=""),
  dict(name="v" ,type="f4" ,units="m/s" ,standard_name="" ,long_name=""),
  dict(name="w" ,type="f4" ,units="m/s" ,standard_name="" ,long_name=""),
]

spectravar = [
  dict(name="e" ,type="f4" ,units="watts" ,standard_name="" ,long_name=""),
]


Intput = dict(
  name="test2",
  folder=None,
  localOnly=False,
  autoUpload=True,
  bucket="merac-dev",
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
      dict(name="b" ,type="f4" ,dimensions=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="lat" ,type="f8" ,dimensions=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="lng" ,type="f8" ,dimensions=["nnode"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="elem" ,type="i4" ,dimensions=["nelem"] ,units="m" ,standard_name="" ,long_name=""),
      dict(name="time" ,type="f8" ,dimensions=["ntime"] ,units="hours since 1970-01-01 00:00:00.0" ,calendar="gregorian" ,standard_name="" ,long_name=""),
    ]),
  nca = dict(
    size=1.0,
    dimensions = [
      dict(name="ntime" ,value=ntime),
      dict(name="nnode" ,value=nnode),
      dict(name="nspectra" ,value=nspectra),
      dict(name="nfreq" ,value=nfreq),
      dict(name="ndir" ,value=ndir),
    ],
    groups=[
      dict(name="s" ,dimensions=["ntime", "nnode"] ,variables=vars),
      dict(name="t" ,dimensions=["nnode" ,"ntime"] ,variables=vars),
      dict(name="ss" ,dimensions=["ntime", "nspectra", "nfreq", "ndir"] ,variables=spectravar),
      dict(name="st" ,dimensions=["nspectra" ,"ntime", "nfreq", "ndir"] ,variables=spectravar)
    ]
  )
)



