
__all__ = [ "Status" ]

from Gaugi import EnumStringification


class Status ( EnumStringification ):

  BROKEN = "broken"
  FAILED = "failed"
  KILLED = "killed"
  DONE = "done"
  REGISTERED = "registered"
  TESTING = "testing"
  ASSIGNED = "assigned"
  ACTIVATED = "activated"
  STARTING = "starting"
  RUNNING  = "running"






