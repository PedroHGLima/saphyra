
__all__ = ['ReferenceReader']


from Gaugi import Logger


class ReferenceReader( Logger ):

  def __init__( self, **kw ):
    Logger.__init__(self, kw)
    self._obj = None

  def load( self, ofile ):

    from Gaugi import load
    raw = load( ofile )
    # get the file version
    version = raw['__version']

    # the current file version
    if version == 1:
      from saphyra import Reference_v1
      self._obj = Reference_v1()
      self._obj.fromRawObj( raw )
    else:
      # error because the file does not exist
      self._logger.fatal( 'File version (%d) not supported in (%s)', version, ofile)

    # return the list of keras models
    return self._obj
    

  def save(self, obj, ofile):
    obj.save(ofile)


  def get_object(self):
    return self._obj







