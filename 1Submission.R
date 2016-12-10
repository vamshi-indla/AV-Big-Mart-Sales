### 5. Submission
################################
#
# preparing the final submissions
#
################################

final <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction)

names(final12) <- c("Item_Identifier",
                    "Outlet_Identifier",
                    "Item_Outlet_Sales")

write.csv(final12, file="final.csv", row.names=FALSE, quote = FALSE)
