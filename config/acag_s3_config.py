"""SatPM V6.GL.02.04 on AWS Open Data (bucket v6.gl.02.04, us-west-2).

Monthly global grids at 0.1°: ``V6.GL.02.04-0p10/GL/Monthly/...``
Downloads often require AWS credentials and requester-pays billing; see README.
"""

S3_BUCKET: str = "v6.gl.02.04"
S3_REGION: str = "us-west-2"

# Global monthly PM2.5, 0.1° — one NetCDF per calendar month
S3_KEY_TEMPLATE: str = (
    "V6.GL.02.04-0p10/GL/Monthly/{year}/"
    "V6GL02.04.CNNPM25.0p10.GL.{ym}-{ym}.nc"
)
