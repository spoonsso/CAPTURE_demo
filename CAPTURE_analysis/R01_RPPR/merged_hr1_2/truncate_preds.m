load("predictions.mat")

length = 50000;

animalID = animalID(1:length,:);
fn = fieldnames(predictions);
for i=1:numel(fn)-1
    predictions.(fn{i})=predictions.(fn{i})(1:length,:);
end
predictions.sampleID = predictions.sampleID(:,1:length);

clear length fn 
save('truncated_preds.mat','-v7.3')