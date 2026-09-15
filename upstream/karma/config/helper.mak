
printvars:
	$(foreach V, \
		$(sort $(.VARIABLES)), \
		$(if \
			$(filter-out environment% default automatic, $(origin $V)), \
			$(warning $V=$($V) ($(value $V))) \
		) \
	)
	
echo_% :
#	@echo $* = '${$*}'
	@echo '${$*}'

# print all variables separated by "!!"
# split based on the separator "!!"
# then split based on "=" and get key/value pair.
echo_all:
	@echo $(foreach V, \
		$(sort $(.VARIABLES)), \
		$(if \
			$(filter-out environment% default automatic, $(origin $V)), \
			$V=$($V)!! \
		) \
	)
