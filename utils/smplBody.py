import os
import torch
from human_body_prior.body_model.body_model import BodyModel as BM


class BodyModel(torch.nn.Module):
    def __init__(self, support_dir, smplx = False):
        super().__init__()
        subject_gender = "male"
        bm_fname = os.path.join(
            support_dir, "{}/{}/model.npz".format('smplx' if smplx else 'smplh', subject_gender)
        )
        dmpl_fname = os.path.join(
            support_dir, "dmpls/{}/model.npz".format(subject_gender)
        ) if not smplx else None
        num_betas = 16 if not smplx else 10  # number of body parameters
        num_dmpls = 8 if not smplx else None  # number of DMPL parameters
        self.body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
            model_type='smplx' if smplx else'smplh' # From XRZ
        )

    def forward(self, body_params):  # body_params:{pose_body:(N, 63), root_orient:(N, 3)}
        # with torch.no_grad():
        body_pose = self.body_model(
            **{
                k: v
                for k, v in body_params.items()
                if k in ["pose_body", "trans", "root_orient"]
            }
        )
        # body_pose.v: (batch, 6890, 3)  vertice
        # body_pose.f: (13776, 3)
        # body_pose.Jtr: (batch, 52, 3) joints location
        # body_pose.full_pose: (batch, 156)
        return body_pose
