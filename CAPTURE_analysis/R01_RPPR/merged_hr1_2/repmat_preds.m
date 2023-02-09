load("predictions.mat")

d = 8;

animalID = repmat(animalID,d,1);
fn = fieldnames(predictions);
for i=1:numel(fn)-1
    predictions.(fn{i})=repmat(predictions.(fn{i}),d,1);
end
predictions.sampleID = repmat(predictions.sampleID,1,d);

clear d fn 
save('duped_preds.mat','-v7.3')