import xml.sax

class MovieHandler( xml.sax.ContentHandler ):
   def __init__(self):
      self.CurrentData = ""
      self.unique_id = ""
      self.asin = ""
      self.product_name = ""
      self.product_type = ""
      self.helpful = ""
      self.rating = ""
      self.title = ""
      self.date = ""
      self.reviewer = ""
      self.reviever_location = ""
      self.review_text = ""

   # Call when an element starts
   def startElement(self, tag, attributes):
      self.CurrentData = tag
      if tag == "review":
         print "*****Review*****"
         #title = attributes["review"]
         #print "Title:", title

   # Call when an elements ends
   def endElement(self, tag):
      if self.CurrentData == "unique_id":
         print "unique_id:", self.unique_id
      elif self.CurrentData == "asin":
         print "asin:", self.asin
      elif self.CurrentData == "product_name":
         print "product_name:", self.product_name
      elif self.CurrentData == "product_type":
         print "product_type:", self.product_type
      elif self.CurrentData == "helpful":
         print "helpful:", self.helpful
      elif self.CurrentData == "rating":
         print "rating:", self.rating
      elif self.CurrentData == "title":
         print "title:", self.title
      elif self.CurrentData == "date":
         print "date:", self.date
      elif self.CurrentData == "reviewer":
         print "reviewer:", self.reviewer
      elif self.CurrentData == "reviever_location":
         print "reviever_location:", self.reviever_location
      elif self.CurrentData == "review_text":
         print "review_text:", self.review_text
      else:
         self.CurrentData = ""

   # Call when a character is read
   def characters(self, content):
      if self.CurrentData == "unique_id":
         self.unique_id = content
      elif self.CurrentData == "asin":
         self.asin = content
      elif self.CurrentData == "product_name":
         self.product_name = content
      elif self.CurrentData == "product_type":
         self.product_type = content
      elif self.CurrentData == "helpful":
         self.helpful = content
      elif self.CurrentData == "rating":
         self.rating = content
      elif self.CurrentData == "title":
         self.title = content
      elif self.CurrentData == "date":
         self.date = content
      elif self.CurrentData == "reviewer":
         self.reviewer = content
      elif self.CurrentData == "reviever_location":
         self.reviever_location = content
      elif self.CurrentData == "review_text":
         self.review_text = content

if ( __name__ == "__main__"):
   # create an XMLReader
   parser = xml.sax.make_parser()
   # turn off namepsaces
   parser.setFeature(xml.sax.handler.feature_namespaces, 0)

   # override the default ContextHandler
   Handler = MovieHandler()
   parser.setContentHandler( Handler )

   parser.parse("sorted_data_acl/books/positive_sample.review")