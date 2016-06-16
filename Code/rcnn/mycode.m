% The following code runs the rcnn algorithm for 3 images per class in 60 classes
fpath = '/media/syed/Druid/results'; % path for input images
rcnn_model_file = './data/rcnn_models/voc_2012/rcnn_model_finetuned.mat'; % using the PASCAL VOC 12 pre-trained model
use_gpu = 0; % Choosing CPU Mode
clf;
thresh = -1;
for	myclass = 1:60 % Classes
	for num = 1:3 % Images
	im = imread('./examples/images/voc12/mycalss/num (2).jpg'); %input
    fprintf('Initializing R-CNN model (this might take a little while)\n');
    rcnn_model = rcnn_load_model(rcnn_model_file, use_gpu);
    fprintf('done\n');
    th = tic;
    dets = rcnn_detect(im, rcnn_model, thresh); % find the proposals....
    fprintf('Total %d-class detection time: %.3fs\n', ...
        length(rcnn_model.classes), toc(th));

    all_dets = [];
    for i = 1:length(dets)
      all_dets = cat(1, all_dets, ...
          [i * ones(size(dets{i}, 1), 1) dets{i}]);
    end

    [~, ord] = sort(all_dets(:,end), 'descend');
    for i = 1:length(ord)
      score = all_dets(ord(i), end);
      if score < 0
        break;
      end
      cls = rcnn_model.classes{all_dets(ord(i), 1)};
      myout = figure;
      showboxes(im, all_dets(ord(i), 2:5));
      title(sprintf('det #%d: %s score = %.3f', ...
          i, cls, score));
      drawnow;
      saveas(myout,fullfile(fpath,'2(2)'),'jpg') % Saving the output
    end
end