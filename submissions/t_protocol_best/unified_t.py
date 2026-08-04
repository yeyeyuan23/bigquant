"""Frozen original Temporal-deep expert for AIStudio."""

from __future__ import annotations

import base64
import hashlib
import io
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

KEY_COLUMNS = ("date", "instrument")
FROZEN_CANDIDATE_SPEC = {'candidate_ids': ('FR-001', 'FR-002', 'FR-003', 'FR-004', 'FR-006', 'FR-007', 'FR-010', 'FR-013', 'FR-015', 'HF-001', 'HF-002', 'HF-003', 'HF-004', 'HF-005', 'HF-006', 'HF-007', 'HF-008', 'HF-009', 'HF-010', 'HF-011', 'HF-012', 'HF-013', 'HF-014', 'HF-015', 'HF-016', 'HF-017', 'HF-018', 'HF-019', 'HF-020', 'HF-021', 'HF-022', 'HF-023', 'HF-024', 'HF-025', 'HF-026', 'HF-027', 'HF-028', 'HF-029', 'HF-030', 'HF-031', 'HF-032', 'HF-033', 'HF-034', 'HF-035', 'HF-036', 'HF-037', 'HF-038', 'HF-039', 'HF-040', 'HF-041', 'HF-042', 'HF-043', 'HF-044', 'HF-045', 'HF-046', 'HF-047', 'HF-048', 'HF-049', 'HF-050', 'HF-051', 'HF-052', 'HF-053', 'HF-054', 'HF-055', 'HF-056', 'HF-057', 'HF-058', 'HF-059', 'HF-060', 'HF-061', 'HF-062', 'HF-063', 'HF-064', 'HF-065', 'HF-066', 'HF-067', 'HF-068', 'HF-069', 'HF-070', 'HF-071', 'HF-072', 'HF-073', 'HF-074', 'HF-075', 'HF-076', 'HF-077', 'HF-078', 'HF-079', 'HF-080', 'HF-081', 'HF-082', 'HF-083', 'HF-084', 'HF-085', 'HF-086', 'HF-087', 'HF-088', 'HF-089', 'HF-090', 'HF-091', 'HF-092', 'HF-093', 'HF-094', 'HF-095', 'HF-096', 'HF-097', 'HF-098', 'HF-099', 'HF-100', 'HF-101', 'HF-102', 'HF-103', 'HF-104', 'HF-105', 'HF-106', 'HF-107', 'HF-108', 'HF-109', 'HF-110', 'HF-111', 'HF-112', 'HF-113', 'HF-114', 'HF-115', 'HF-116', 'HF-117', 'HF-118', 'HF-119', 'HF-120', 'HF-121', 'HF-122', 'HF-123', 'HF-124', 'HF-125', 'HF-126', 'HF-127', 'HF-128', 'HF-129', 'HF-130', 'HF-131', 'HF-132', 'HF-133', 'HF-134', 'HF-135', 'HF-136', 'HF-137', 'HF-138', 'HF-139', 'HF-140', 'HF-141', 'HF-142', 'HF-143', 'HF-144', 'HF-145', 'HF-146', 'HF-147', 'HF-148', 'HF-149', 'HF-150', 'HF-151', 'HF-152', 'HF-153', 'HF-154', 'HF-155', 'HF-156', 'HF-157', 'HF-158', 'HF-159', 'HF-160', 'HF-161', 'HF-162', 'HF-163', 'HF-164', 'HF-165', 'HF-166', 'HF-167', 'HF-168', 'HF-169', 'HF-170', 'HF-171', 'HF-172', 'HF-173', 'HF-174', 'HF-175', 'HF-176', 'HF-177', 'HF-178', 'HF-179', 'HF-180', 'HF-181', 'HF-182', 'HF-183', 'HF-184', 'HF-185', 'HF-186', 'HF-187', 'HF-188', 'HF-189', 'HF-190', 'HF-191', 'HF-192', 'HF-193', 'HF-194', 'HF-195', 'HF-196', 'HF-197', 'HF-198', 'HF-199', 'HF-200', 'HF-201', 'HF-202', 'HF-203', 'HF-204', 'HF-205', 'HF-206', 'HF-207', 'HF-208', 'HF-209', 'HF-210', 'HF-211', 'HF-212', 'HF-213', 'HF-214', 'HF-215', 'HF-216', 'HF-217', 'HF-218', 'HF-219', 'HF-220', 'HF-221', 'HF-222', 'HF-223', 'HF-224', 'INT-002', 'INT-004', 'INT-005', 'INT-006', 'INT-007', 'INT-008', 'INT-009', 'INT-010', 'INT-011', 'INT-012', 'INT-013', 'INT-014', 'INT-015', 'INT-016', 'INT-017', 'OB-001', 'OB-002', 'OB-003', 'OB-004', 'OB-005', 'OB-006', 'OB-007', 'OB-008', 'PV-001', 'PV-002', 'PV-003', 'PV-004', 'PV-005', 'PV-006', 'PV-007', 'PV-014', 'PV-020', 'PV-021', 'PV-023', 'PV-024', 'PV-025', 'PV-026', 'PV-027', 'PV-028', 'PV-029', 'PV-030', 'PV-031', 'PV-032', 'PV-033', 'PV-034', 'PV-035', 'PV-036', 'PV-037', 'PV-038', 'PV-039', 'PV-040', 'PV-041', 'PV-042', 'PV-043', 'PV-044', 'PV-045', 'PV-046', 'PV-047', 'PV-048', 'PV-049', 'PV-050', 'PV-051', 'PV-052', 'PV-053', 'PV-054', 'PV-055', 'PV-056', 'PV-057', 'PV-058', 'PV-059', 'PV-060', 'PV-061', 'PV-062', 'PV-063', 'PV-064', 'PV-065', 'PV-068', 'PV-069', 'PV-070', 'PV-071', 'PV-072', 'PV-074', 'PV-075', 'PV-076', 'PV-078', 'PV-079', 'PV-080', 'PV-081', 'PV-082', 'PV-084', 'PV-085', 'PV-086', 'PV-087', 'PV-088', 'PV-089', 'PV-090', 'PV-091', 'PV-092', 'PV-093', 'PV-094', 'PV-095', 'PV-096', 'PV-097', 'PV-098', 'PV-099', 'PV-100', 'PV-101', 'PV-102', 'PV-103', 'PV-104', 'PV-105', 'PV-106', 'PV-107', 'PV-108', 'PV-109', 'PV-110', 'PV-111', 'PV-112', 'PV-113', 'PV-114', 'PV-115', 'PV-116', 'PV-117', 'PV-118', 'PV-119', 'PV-120', 'PV-122', 'PV-123', 'PV-124', 'PV-125', 'PV-126', 'PV-127', 'PV-128', 'PV-129', 'PV-130', 'PV-131', 'PV-132', 'PV-133', 'PV-134', 'PV-135', 'PV-136', 'PV-137', 'PV-138', 'PV-139', 'PV-140', 'PV-141', 'PV-142', 'PV-143', 'PV-144', 'PV-145', 'PV-146', 'PV-147', 'PV-148', 'PV-149', 'PV-150', 'PV-151', 'PV-152', 'PV-153', 'PV-154', 'PV-155', 'PV-156', 'PV-157', 'PV-158', 'PV-159', 'PV-160', 'PV-161', 'PV-162', 'PV-163', 'PV-164', 'PV-165', 'PV-166', 'PV-167', 'PV-168', 'PV-169', 'PV-170', 'PV-171', 'PV-172', 'PV-173', 'PV-174', 'PV-176', 'PV-177', 'PV-178', 'PV-179', 'PV-180', 'PV-181', 'PV-182', 'PV-183', 'PV-184', 'PV-185', 'PV-186', 'PV-187', 'PV-188', 'PV-189', 'PV-190', 'PV-191', 'PV-192', 'PV-193', 'PV-194', 'PV-196', 'PV-197', 'PV-198', 'PV-199', 'PV-200', 'PV-201', 'PV-202', 'PV-203', 'PV-204', 'PV-205', 'PV-206', 'PV-207', 'PV-208', 'PV-209', 'PV-210', 'PV-211', 'PV-212', 'PV-213', 'PV-214', 'PV-215', 'PV-217', 'PV-218', 'PV-219'), 'component_specs': {'HF-005': ('mmt_pm', 1.0, 'cicc'), 'HF-006': ('mmt_last30', 1.0, 'cicc'), 'HF-007': ('mmt_paratio', 1.0, 'cicc'), 'HF-008': ('mmt_am', 1.0, 'cicc'), 'HF-009': ('mmt_between', 1.0, 'cicc'), 'HF-010': ('mmt_ols_corr_sqaure_mean', 1.0, 'cicc'), 'HF-011': ('mmt_ols_corr_mean', 1.0, 'cicc'), 'HF-012': ('mmt_ols_beta_mean', 1.0, 'cicc'), 'HF-013': ('mmt_ols_beta_zscore_last', 1.0, 'cicc'), 'HF-014': ('vol_volume1min', -1.0, 'cicc'), 'HF-015': ('vol_range1min', -1.0, 'cicc'), 'HF-016': ('vol_return1min', -1.0, 'cicc'), 'HF-017': ('shape_skew', -1.0, 'cicc'), 'HF-018': ('shape_kurt', -1.0, 'cicc'), 'HF-019': ('shape_skewVol', -1.0, 'cicc'), 'HF-020': ('shape_kurtVol', -1.0, 'cicc'), 'HF-021': ('liq_amihud_1min', 1.0, 'cicc'), 'HF-022': ('liq_closevol', 1.0, 'cicc'), 'HF-023': ('corr_prv', -1.0, 'cicc'), 'HF-024': ('corr_prvr', -1.0, 'cicc'), 'HF-025': ('corr_pv', -1.0, 'cicc'), 'HF-026': ('corr_pvr', -1.0, 'cicc'), 'HF-027': ('trade_bottom20retRatio', 1.0, 'cicc'), 'HF-028': ('trade_bottom50retRatio', 1.0, 'cicc'), 'HF-029': ('trade_headRatio', 1.0, 'cicc'), 'HF-030': ('trade_tailRatio', -1.0, 'cicc'), 'HF-031': ('trade_top50retRatio', 1.0, 'cicc'), 'HF-037': ('FZ-001', -1.0, 'fangzheng'), 'HF-038': ('FZ-005', -1.0, 'fangzheng'), 'HF-039': ('FZ-009', -1.0, 'fangzheng'), 'HF-040': ('FZ-010', -1.0, 'fangzheng'), 'HF-041': ('FZ-011', -1.0, 'fangzheng'), 'HF-042': ('FZ-014', -1.0, 'fangzheng'), 'HF-043': ('FZ-018', -1.0, 'fangzheng'), 'HF-044': ('FZ-019', -1.0, 'fangzheng'), 'HF-045': ('FZ-020', -1.0, 'fangzheng'), 'HF-046': ('FZ-021', -1.0, 'fangzheng'), 'HF-047': ('FZ-022', -1.0, 'fangzheng'), 'HF-048': ('FZ-023', -1.0, 'fangzheng'), 'HF-049': ('FZ-024', -1.0, 'fangzheng'), 'HF-050': ('FZ-025', -1.0, 'fangzheng'), 'HF-051': ('FZ-028', -1.0, 'fangzheng'), 'HF-052': ('FZ-032', -1.0, 'fangzheng'), 'HF-053': ('FZ-033', -1.0, 'fangzheng'), 'HF-054': ('FZ-039', 1.0, 'fangzheng'), 'HF-055': ('FZ-040', 1.0, 'fangzheng'), 'HF-056': ('FZ-041', 1.0, 'fangzheng'), 'HF-057': ('FZ-042', 1.0, 'fangzheng'), 'HF-058': ('FZ-043', 1.0, 'fangzheng'), 'HF-059': ('FZ-044', 1.0, 'fangzheng'), 'HF-060': ('FZ-045', 1.0, 'fangzheng'), 'HF-061': ('FZ-046', 1.0, 'fangzheng'), 'HF-062': ('FZ-047', 1.0, 'fangzheng'), 'HF-063': ('FZ-064', -1.0, 'fangzheng'), 'HF-064': ('FZ-065', -1.0, 'fangzheng'), 'HF-065': ('FZ-066', -1.0, 'fangzheng'), 'HF-066': ('FZ-067', -1.0, 'fangzheng'), 'HF-067': ('FZ-068', -1.0, 'fangzheng'), 'HF-068': ('FZ-070', -1.0, 'fangzheng'), 'HF-069': ('FZ-071', -1.0, 'fangzheng'), 'HF-070': ('FZ-076', -1.0, 'fangzheng'), 'HF-071': ('FZ-077', -1.0, 'fangzheng'), 'HF-072': ('FZ-078', -1.0, 'fangzheng'), 'HF-073': ('FZ-079', -1.0, 'fangzheng'), 'HF-074': ('FZ-083', -1.0, 'fangzheng'), 'HF-075': ('FZ-085', -1.0, 'fangzheng'), 'HF-076': ('FZ-086', -1.0, 'fangzheng'), 'HF-077': ('FZ-087', -1.0, 'fangzheng'), 'HF-078': ('FZ-088', -1.0, 'fangzheng'), 'HF-079': ('CICC-011', 1.0, 'cicc'), 'HF-080': ('CICC-012', 1.0, 'cicc'), 'HF-081': ('CICC-013', 1.0, 'cicc'), 'HF-082': ('CICC-014', 1.0, 'cicc'), 'HF-083': ('CICC-018', 1.0, 'cicc'), 'HF-084': ('CICC-019', 1.0, 'cicc'), 'HF-085': ('CICC-020', 1.0, 'cicc'), 'HF-086': ('CICC-024', 1.0, 'cicc'), 'HF-087': ('CICC-027', 1.0, 'cicc'), 'HF-088': ('CICC-042', 1.0, 'cicc'), 'HF-089': ('CICC-076', 1.0, 'cicc'), 'HF-090': ('CICC-078', 1.0, 'cicc'), 'HF-091': ('CICC-079', 1.0, 'cicc'), 'HF-092': ('HAITONG-0038', 1.0, 'haitong'), 'HF-093': ('HAITONG-0040', 1.0, 'haitong'), 'HF-094': ('HAITONG-0043', 1.0, 'haitong'), 'HF-095': ('HAITONG-0045', 1.0, 'haitong'), 'HF-096': ('HAITONG-0047', 1.0, 'haitong'), 'HF-097': ('HAITONG-0048', 1.0, 'haitong'), 'HF-098': ('HAITONG-0050', 1.0, 'haitong'), 'HF-099': ('HAITONG-0052', 1.0, 'haitong'), 'HF-100': ('HAITONG-0072', 1.0, 'haitong'), 'HF-101': ('HAITONG-0076', 1.0, 'haitong'), 'HF-102': ('HAITONG-0080', 1.0, 'haitong'), 'HF-103': ('HAITONG-0132', 1.0, 'haitong'), 'HF-105': ('CJ-G006-V01', 1.0, 'changjiang'), 'HF-106': ('CJ-G006-V02', 1.0, 'changjiang'), 'HF-107': ('CJ-G006-V03', 1.0, 'changjiang'), 'HF-108': ('CJ-G007-V01', 1.0, 'changjiang'), 'HF-109': ('CJ-G007-V02', 1.0, 'changjiang'), 'HF-110': ('CJ-G007-V03', 1.0, 'changjiang'), 'HF-111': ('CJ-G008-V05', 1.0, 'changjiang'), 'HF-112': ('CJ-G011-V01', 1.0, 'changjiang'), 'HF-113': ('CJ-G011-V03', 1.0, 'changjiang'), 'HF-114': ('CJ-G012-V03', 1.0, 'changjiang'), 'HF-115': ('CJ-G013-V01', 1.0, 'changjiang'), 'HF-116': ('CJ-G013-V02', 1.0, 'changjiang'), 'HF-117': ('CJ-G013-V03', 1.0, 'changjiang'), 'HF-118': ('CJ-G014-V01', 1.0, 'changjiang'), 'HF-119': ('CJ-G014-V02', 1.0, 'changjiang'), 'HF-120': ('CJ-G014-V03', 1.0, 'changjiang'), 'HF-121': ('CJ-G014-V04', 1.0, 'changjiang'), 'HF-122': ('CJ-G015-V01', 1.0, 'changjiang'), 'HF-123': ('CJ-G015-V02', 1.0, 'changjiang'), 'HF-124': ('CJ-G015-V03', 1.0, 'changjiang'), 'HF-125': ('CJ-G015-V04', 1.0, 'changjiang'), 'HF-126': ('CJ-G020-V02', 1.0, 'changjiang'), 'HF-127': ('CJ-G025-V01', 1.0, 'changjiang'), 'HF-128': ('CJ-G028-V02', 1.0, 'changjiang'), 'HF-129': ('CJ-G029-V01', 1.0, 'changjiang'), 'HF-130': ('CJ-G029-V04', 1.0, 'changjiang'), 'HF-131': ('CJ-G032-V01', 1.0, 'changjiang'), 'HF-132': ('CJ-G033-V01', 1.0, 'changjiang'), 'HF-133': ('CJ-G034-V01', 1.0, 'changjiang'), 'HF-134': ('CJ-G035-V01', 1.0, 'changjiang'), 'HF-135': ('CJ-G039-V01', 1.0, 'changjiang'), 'HF-136': ('CJ-G039-V02', 1.0, 'changjiang'), 'HF-137': ('CJ-G039-V03', 1.0, 'changjiang'), 'HF-138': ('CJ-G039-V04', 1.0, 'changjiang'), 'HF-139': ('CJ-G039-V05', 1.0, 'changjiang'), 'HF-140': ('CJ-G039-V06', 1.0, 'changjiang'), 'HF-141': ('CJ-G039-V07', 1.0, 'changjiang'), 'HF-142': ('CJ-G039-V08', 1.0, 'changjiang'), 'HF-143': ('CJ-G039-V09', 1.0, 'changjiang'), 'HF-144': ('CJ-G039-V10', 1.0, 'changjiang'), 'HF-145': ('CJ-G039-V11', 1.0, 'changjiang'), 'HF-146': ('CJ-G039-V12', 1.0, 'changjiang'), 'HF-147': ('CJ-G039-V15', 1.0, 'changjiang'), 'HF-148': ('CJ-G039-V18', 1.0, 'changjiang'), 'HF-149': ('CJ-G040-V01', 1.0, 'changjiang'), 'HF-150': ('CJ-G042-V01', 1.0, 'changjiang'), 'HF-151': ('CJ-G042-V02', 1.0, 'changjiang'), 'HF-152': ('CJ-G042-V03', 1.0, 'changjiang'), 'HF-153': ('CJ-G042-V04', 1.0, 'changjiang'), 'HF-154': ('CJ-G042-V05', 1.0, 'changjiang'), 'HF-155': ('CJ-G042-V06', 1.0, 'changjiang'), 'HF-156': ('CJ-G042-V07', 1.0, 'changjiang'), 'HF-157': ('CJ-G042-V08', 1.0, 'changjiang'), 'HF-158': ('CJ-G042-V09', 1.0, 'changjiang'), 'HF-159': ('CJ-G044-V01', 1.0, 'changjiang'), 'HF-160': ('CJ-G044-V02', 1.0, 'changjiang'), 'HF-161': ('CJ-G044-V03', 1.0, 'changjiang'), 'HF-162': ('CJ-G044-V04', 1.0, 'changjiang'), 'HF-163': ('CJ-G044-V05', 1.0, 'changjiang'), 'HF-164': ('CJ-G046-V01', 1.0, 'changjiang'), 'HF-165': ('CJ-G046-V02', 1.0, 'changjiang'), 'HF-166': ('CJ-G046-V03', 1.0, 'changjiang'), 'HF-167': ('CJ-G046-V04', 1.0, 'changjiang'), 'HF-168': ('CJ-G046-V05', 1.0, 'changjiang'), 'HF-169': ('CJ-G048-V01', 1.0, 'changjiang'), 'HF-170': ('CJ-G049-V01', 1.0, 'changjiang'), 'HF-171': ('CJ-G050-V01', 1.0, 'changjiang'), 'HF-172': ('CJ-G051-V01', 1.0, 'changjiang'), 'HF-173': ('CJ-G051-V02', 1.0, 'changjiang'), 'HF-174': ('CJ-G051-V03', 1.0, 'changjiang'), 'HF-175': ('CJ-G051-V04', 1.0, 'changjiang'), 'HF-176': ('CJ-G051-V05', 1.0, 'changjiang'), 'HF-177': ('CJ-G058-V01', 1.0, 'changjiang'), 'HF-178': ('CJ-G058-V02', 1.0, 'changjiang'), 'HF-179': ('CJ-G060-V01', 1.0, 'changjiang'), 'HF-180': ('CJ-G063-V01', 1.0, 'changjiang'), 'HF-181': ('CJ-G063-V02', 1.0, 'changjiang'), 'HF-182': ('CJ-G064-V01', 1.0, 'changjiang'), 'HF-183': ('CJ-G064-V02', 1.0, 'changjiang'), 'HF-184': ('CJ-G064-V03', 1.0, 'changjiang'), 'HF-185': ('CJ-G065-V02', 1.0, 'changjiang'), 'HF-186': ('CJ-G065-V03', 1.0, 'changjiang'), 'HF-187': ('CJ-G066-V01', 1.0, 'changjiang'), 'HF-188': ('CJ-G067-V01', 1.0, 'changjiang'), 'HF-189': ('CJ-G068-V01', 1.0, 'changjiang'), 'HF-190': ('CJ-G071-V01', 1.0, 'changjiang'), 'HF-191': ('CJ-G072-V01', 1.0, 'changjiang'), 'HF-192': ('CJ-G074-V01', 1.0, 'changjiang'), 'HF-193': ('CJ-G075-V01', 1.0, 'changjiang'), 'HF-194': ('CJ-G077-V01', 1.0, 'changjiang'), 'HF-195': ('CJ-G079-V01', 1.0, 'changjiang'), 'HF-196': ('CJ-G080-V01', 1.0, 'changjiang'), 'HF-197': ('CJ-G083-V01', 1.0, 'changjiang'), 'HF-198': ('CJ-G084-V01', 1.0, 'changjiang'), 'HF-199': ('CJ-G085-V01', 1.0, 'changjiang'), 'HF-200': ('CJ-G086-V01', 1.0, 'changjiang'), 'HF-201': ('CJ-G087-V01', 1.0, 'changjiang'), 'HF-202': ('CJ-G088-V01', 1.0, 'changjiang'), 'HF-203': ('CJ-G090-V01', 1.0, 'changjiang'), 'HF-204': ('CJ-G091-V01', 1.0, 'changjiang'), 'HF-205': ('CJ-G092-V01', 1.0, 'changjiang'), 'HF-206': ('CJ-G097-V01', 1.0, 'changjiang'), 'HF-207': ('CJ-G098-V01', 1.0, 'changjiang'), 'HF-208': ('CJ-G099-V01', 1.0, 'changjiang'), 'HF-209': ('CJ-G100-V01', 1.0, 'changjiang'), 'HF-210': ('CJ-G101-V01', 1.0, 'changjiang'), 'HF-211': ('CJ-G102-V01', 1.0, 'changjiang'), 'HF-212': ('CJ-G103-V01', 1.0, 'changjiang'), 'HF-213': ('CJ-G103-V02', 1.0, 'changjiang'), 'HF-214': ('CJ-G103-V03', 1.0, 'changjiang'), 'HF-215': ('CJ-G103-V04', 1.0, 'changjiang'), 'HF-216': ('CJ-G104-V02', 1.0, 'changjiang'), 'HF-217': ('CJ-G104-V03', 1.0, 'changjiang'), 'HF-218': ('CJ-G104-V04', 1.0, 'changjiang'), 'HF-219': ('CJ-G105-V03', 1.0, 'changjiang'), 'HF-220': ('CJ-G105-V07', 1.0, 'changjiang'), 'HF-221': ('CJ-G105-V13', 1.0, 'changjiang'), 'HF-222': ('CJ-G106-V01', 1.0, 'changjiang'), 'HF-223': ('CJ-G107-V02', 1.0, 'changjiang'), 'HF-224': ('CJ-G122-V01', 1.0, 'changjiang'), 'INT-005': ('FZ-015', -1.0, 'fangzheng'), 'INT-006': ('FZ-029', -1.0, 'fangzheng'), 'INT-007': ('FZ-048', 1.0, 'fangzheng'), 'INT-008': ('FZ-063', -1.0, 'fangzheng'), 'INT-009': ('FZ-084', -1.0, 'fangzheng'), 'INT-010': ('FZ-090', -1.0, 'fangzheng'), 'INT-011': ('FZ-091', -1.0, 'fangzheng'), 'INT-012': ('FZ-095', -1.0, 'fangzheng'), 'INT-013': ('FZ-096', -1.0, 'fangzheng'), 'INT-014': ('FZ-104', -1.0, 'fangzheng'), 'INT-015': ('FZ-107', -1.0, 'fangzheng'), 'INT-016': ('FZ-108', -1.0, 'fangzheng'), 'INT-017': ('FZ-109', -1.0, 'fangzheng'), 'OB-007': ('HAITONG-0145', 1.0, 'haitong'), 'PV-024': ('FZ-049', -1.0, 'fangzheng'), 'PV-025': ('FZ-050', -1.0, 'fangzheng'), 'PV-026': ('FZ-051', -1.0, 'fangzheng'), 'PV-027': ('FZ-052', -1.0, 'fangzheng'), 'PV-028': ('FZ-053', -1.0, 'fangzheng'), 'PV-029': ('FZ-054', -1.0, 'fangzheng'), 'PV-030': ('FZ-055', -1.0, 'fangzheng'), 'PV-031': ('FZ-056', -1.0, 'fangzheng'), 'PV-032': ('FZ-057', -1.0, 'fangzheng'), 'PV-033': ('FZ-058', -1.0, 'fangzheng'), 'PV-034': ('FZ-059', -1.0, 'fangzheng'), 'PV-035': ('FZ-060', -1.0, 'fangzheng'), 'PV-036': ('FZ-061', -1.0, 'fangzheng'), 'PV-037': ('FZ-062', -1.0, 'fangzheng'), 'PV-038': ('FZ-089', -1.0, 'fangzheng'), 'PV-039': ('FZ-092', -1.0, 'fangzheng'), 'PV-040': ('FZ-093', -1.0, 'fangzheng'), 'PV-041': ('FZ-099', -1.0, 'fangzheng'), 'PV-042': ('FZ-100', -1.0, 'fangzheng'), 'PV-043': ('FZ-101', -1.0, 'fangzheng'), 'PV-044': ('FZ-103', -1.0, 'fangzheng'), 'PV-202': ('HAITONG-0022', 1.0, 'haitong'), 'PV-203': ('HAITONG-0023', 1.0, 'haitong'), 'PV-204': ('HAITONG-0024', 1.0, 'haitong'), 'PV-205': ('HAITONG-0025', 1.0, 'haitong'), 'PV-206': ('HAITONG-0026', 1.0, 'haitong'), 'PV-207': ('HAITONG-0027', 1.0, 'haitong'), 'PV-208': ('HAITONG-0036', 1.0, 'haitong'), 'PV-209': ('HAITONG-0037', 1.0, 'haitong'), 'PV-210': ('HAITONG-0100', 1.0, 'haitong'), 'PV-211': ('HAITONG-0101', 1.0, 'haitong'), 'PV-212': ('HAITONG-0102', 1.0, 'haitong'), 'PV-213': ('HAITONG-0103', 1.0, 'haitong'), 'PV-214': ('HAITONG-0104', 1.0, 'haitong'), 'PV-215': ('HAITONG-0105', 1.0, 'haitong'), 'PV-217': ('CJ-G020-V03', 1.0, 'changjiang'), 'PV-218': ('CJ-G062-V03', 1.0, 'changjiang'), 'PV-219': ('CJ-G112-V01', 1.0, 'changjiang')}, 'gtja_specs': {'PV-045': ('gtja_001', 1.0), 'PV-046': ('gtja_002', 1.0), 'PV-047': ('gtja_003', 1.0), 'PV-048': ('gtja_004', 1.0), 'PV-049': ('gtja_005', 1.0), 'PV-050': ('gtja_006', 1.0), 'PV-051': ('gtja_007', 1.0), 'PV-052': ('gtja_008', 1.0), 'PV-053': ('gtja_009', 1.0), 'PV-054': ('gtja_011', 1.0), 'PV-055': ('gtja_012', 1.0), 'PV-056': ('gtja_013', 1.0), 'PV-057': ('gtja_014', 1.0), 'PV-058': ('gtja_015', 1.0), 'PV-059': ('gtja_016', 1.0), 'PV-060': ('gtja_018', 1.0), 'PV-061': ('gtja_020', 1.0), 'PV-062': ('gtja_021', 1.0), 'PV-063': ('gtja_022', 1.0), 'PV-064': ('gtja_023', 1.0), 'PV-065': ('gtja_024', 1.0), 'PV-068': ('gtja_027', 1.0), 'PV-069': ('gtja_028', 1.0), 'PV-070': ('gtja_029', 1.0), 'PV-071': ('gtja_031', 1.0), 'PV-072': ('gtja_032', 1.0), 'PV-074': ('gtja_035', 1.0), 'PV-075': ('gtja_037', 1.0), 'PV-076': ('gtja_038', 1.0), 'PV-078': ('gtja_040', 1.0), 'PV-079': ('gtja_041', 1.0), 'PV-080': ('gtja_042', 1.0), 'PV-081': ('gtja_043', 1.0), 'PV-082': ('gtja_044', 1.0), 'PV-084': ('gtja_047', 1.0), 'PV-085': ('gtja_048', 1.0), 'PV-086': ('gtja_049', 1.0), 'PV-087': ('gtja_052', 1.0), 'PV-088': ('gtja_053', 1.0), 'PV-089': ('gtja_055', 1.0), 'PV-090': ('gtja_056', 1.0), 'PV-091': ('gtja_057', 1.0), 'PV-092': ('gtja_058', 1.0), 'PV-093': ('gtja_059', 1.0), 'PV-094': ('gtja_060', 1.0), 'PV-095': ('gtja_061', 1.0), 'PV-096': ('gtja_062', 1.0), 'PV-097': ('gtja_064', 1.0), 'PV-098': ('gtja_065', 1.0), 'PV-099': ('gtja_067', 1.0), 'PV-100': ('gtja_068', 1.0), 'PV-101': ('gtja_069', 1.0), 'PV-102': ('gtja_070', 1.0), 'PV-103': ('gtja_071', 1.0), 'PV-104': ('gtja_073', 1.0), 'PV-105': ('gtja_074', 1.0), 'PV-106': ('gtja_076', 1.0), 'PV-107': ('gtja_077', 1.0), 'PV-108': ('gtja_078', 1.0), 'PV-109': ('gtja_079', 1.0), 'PV-110': ('gtja_080', 1.0), 'PV-111': ('gtja_081', 1.0), 'PV-112': ('gtja_083', 1.0), 'PV-113': ('gtja_084', 1.0), 'PV-114': ('gtja_085', 1.0), 'PV-115': ('gtja_086', 1.0), 'PV-116': ('gtja_087', 1.0), 'PV-117': ('gtja_088', 1.0), 'PV-118': ('gtja_089', 1.0), 'PV-119': ('gtja_090', 1.0), 'PV-120': ('gtja_091', 1.0), 'PV-122': ('gtja_093', 1.0), 'PV-123': ('gtja_094', 1.0), 'PV-124': ('gtja_095', 1.0), 'PV-125': ('gtja_097', 1.0), 'PV-126': ('gtja_098', 1.0), 'PV-127': ('gtja_099', 1.0), 'PV-128': ('gtja_100', 1.0), 'PV-129': ('gtja_101', 1.0), 'PV-130': ('gtja_102', 1.0), 'PV-131': ('gtja_103', 1.0), 'PV-132': ('gtja_104', 1.0), 'PV-133': ('gtja_105', 1.0), 'PV-134': ('gtja_106', 1.0), 'PV-135': ('gtja_107', 1.0), 'PV-136': ('gtja_108', 1.0), 'PV-137': ('gtja_109', 1.0), 'PV-138': ('gtja_110', 1.0), 'PV-139': ('gtja_111', 1.0), 'PV-140': ('gtja_112', 1.0), 'PV-141': ('gtja_113', 1.0), 'PV-142': ('gtja_114', 1.0), 'PV-143': ('gtja_115', 1.0), 'PV-144': ('gtja_116', 1.0), 'PV-145': ('gtja_117', 1.0), 'PV-146': ('gtja_118', 1.0), 'PV-147': ('gtja_119', 1.0), 'PV-148': ('gtja_120', 1.0), 'PV-149': ('gtja_121', 1.0), 'PV-150': ('gtja_122', 1.0), 'PV-151': ('gtja_123', 1.0), 'PV-152': ('gtja_124', 1.0), 'PV-153': ('gtja_125', 1.0), 'PV-154': ('gtja_126', 1.0), 'PV-155': ('gtja_128', 1.0), 'PV-156': ('gtja_129', 1.0), 'PV-157': ('gtja_130', 1.0), 'PV-158': ('gtja_131', 1.0), 'PV-159': ('gtja_132', 1.0), 'PV-160': ('gtja_133', 1.0), 'PV-161': ('gtja_134', 1.0), 'PV-162': ('gtja_135', 1.0), 'PV-163': ('gtja_136', 1.0), 'PV-164': ('gtja_137', 1.0), 'PV-165': ('gtja_139', 1.0), 'PV-166': ('gtja_140', 1.0), 'PV-167': ('gtja_141', 1.0), 'PV-168': ('gtja_142', 1.0), 'PV-169': ('gtja_145', 1.0), 'PV-170': ('gtja_147', 1.0), 'PV-171': ('gtja_148', 1.0), 'PV-172': ('gtja_150', 1.0), 'PV-173': ('gtja_151', 1.0), 'PV-174': ('gtja_152', 1.0), 'PV-176': ('gtja_155', 1.0), 'PV-177': ('gtja_156', 1.0), 'PV-178': ('gtja_157', 1.0), 'PV-179': ('gtja_158', 1.0), 'PV-180': ('gtja_159', 1.0), 'PV-181': ('gtja_160', 1.0), 'PV-182': ('gtja_162', 1.0), 'PV-183': ('gtja_163', 1.0), 'PV-184': ('gtja_164', 1.0), 'PV-185': ('gtja_167', 1.0), 'PV-186': ('gtja_168', 1.0), 'PV-187': ('gtja_169', 1.0), 'PV-188': ('gtja_170', 1.0), 'PV-189': ('gtja_171', 1.0), 'PV-190': ('gtja_172', 1.0), 'PV-191': ('gtja_176', 1.0), 'PV-192': ('gtja_177', 1.0), 'PV-193': ('gtja_178', 1.0), 'PV-194': ('gtja_179', 1.0), 'PV-196': ('gtja_185', 1.0), 'PV-197': ('gtja_186', 1.0), 'PV-198': ('gtja_188', 1.0), 'PV-199': ('gtja_189', 1.0), 'PV-200': ('gtja_190', 1.0), 'PV-201': ('gtja_191', 1.0)}}
FROZEN_MANIFEST = {
    "route": "T_protocol_best",
    "model": "temporal_deep",
    "checkpoint_sha256": "d6d493a693390f59fb7798822f6ac7f419ef21e24d8f375ec14bbd1ab2267352",
    "checkpoint_training_end": "2023-06-30",
    "selection_evidence": "Early-stopped after 6/18 screen configs; selected on 2023 local OOS only",
    "training": False,
    "history_calendar_days": 183,
    "lookback_trading_days": 60,
    "candidate_count": 454,
    "allowed_sources": ["bar1m", "financial", "stock_pool_keys"],
    "evidence_boundary": "Frozen original T inference; not T-residual and not a platform score.",
}


def _load_checkpoint(torch):
    import zlib
    from unified_t_weights import CHECKPOINT

    compressed = base64.b85decode(b"".join(CHECKPOINT["b85"]))
    if len(compressed) != CHECKPOINT["compressed_bytes"]:
        raise RuntimeError("T compressed checkpoint byte-count mismatch")
    if hashlib.sha256(compressed).hexdigest() != CHECKPOINT["compressed_sha256"]:
        raise RuntimeError("T compressed checkpoint hash mismatch")
    payload = zlib.decompress(compressed)
    if len(payload) != CHECKPOINT["bytes"]:
        raise RuntimeError("T checkpoint byte-count mismatch")
    if hashlib.sha256(payload).hexdigest() != CHECKPOINT["sha256"]:
        raise RuntimeError("T checkpoint hash mismatch")
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


def _rank(values, pd):
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series.notna() & series.abs().lt(float("inf"))
    if int(finite.sum()) < 2:
        raise RuntimeError("T produced fewer than two finite cross-sectional predictions")
    ranked = pd.Series(0.0, index=series.index, dtype="float64")
    ranked.loc[finite] = (
        series.loc[finite].rank(method="average", pct=True).to_numpy(dtype="float64")
        * 2.0
        - 1.0
    )
    return ranked.to_numpy(dtype="float64")


def main(datasources, start_date, end_date):
    import gc
    import numpy as np
    import pandas as pd
    import torch

    if "bar1m" not in datasources or "financial" not in datasources:
        raise KeyError("datasources must contain bar1m and financial")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date precedes start_date")

    torch.manual_seed(20260803)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260803)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = _load_checkpoint(torch)
    config = CandidateTemporalConfig(**payload["config"])
    model = CandidateTemporalNetwork(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()

    candidate_ids = tuple(FROZEN_CANDIDATE_SPEC["candidate_ids"])
    panel = build_candidate454(
        datasources,
        start,
        end,
        candidate_spec=FROZEN_CANDIDATE_SPEC,
        return_history=True,
    )
    if panel.empty or panel.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("Candidate454 history panel is empty or duplicated")
    missing = sorted(set(candidate_ids).difference(panel.columns))
    if missing:
        raise RuntimeError(f"Candidate454 schema mismatch: {missing}")

    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    history_days = pd.DatetimeIndex(sorted(panel["date"].unique()))
    requested_days = history_days[(history_days >= start) & (history_days <= end)]
    if len(requested_days) == 0:
        raise RuntimeError("no requested trading dates in Candidate454 panel")
    by_day = {
        day: block.set_index("instrument").sort_index()
        for day, block in panel.groupby("date", sort=True)
    }

    outputs = []
    with torch.inference_mode():
        for day in requested_days:
            current = by_day[pd.Timestamp(day)]
            instruments = tuple(current.index.astype(str))
            position = history_days.get_loc(day)
            window_days = history_days[
                max(0, position - config.lookback + 1) : position + 1
            ]
            if len(window_days) < config.lookback:
                raise RuntimeError(
                    f"insufficient T history for {day.date()}: {len(window_days)} days"
                )
            window = np.stack(
                [
                    by_day[past]
                    .reindex(instruments)
                    .loc[:, list(candidate_ids)]
                    .to_numpy(dtype="float32", copy=True)
                    for past in window_days
                ],
                axis=1,
            )
            values = torch.from_numpy(window).unsqueeze(0).to(device)
            observed = torch.isfinite(values)
            stocks = torch.ones(
                (1, len(instruments)), dtype=torch.bool, device=device
            )
            raw = model(values, observed, stocks).squeeze(0).float().cpu().numpy()
            outputs.append(
                pd.DataFrame(
                    {
                        "date": pd.Timestamp(day),
                        "instrument": instruments,
                        "factor": _rank(raw, pd),
                    }
                )
            )
            del window, values, observed, stocks, raw
            gc.collect()

    output = (
        pd.concat(outputs, ignore_index=True)
        .sort_values(list(KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    output = output.loc[
        output["date"].between(start, end), ["date", "instrument", "factor"]
    ]
    if output.empty or output.duplicated(list(KEY_COLUMNS)).any():
        raise RuntimeError("T output is empty or duplicated")
    if not np.isfinite(output["factor"].to_numpy()).all():
        raise RuntimeError("T output contains non-finite values")
    if output.groupby("date")["factor"].nunique().le(1).any():
        raise RuntimeError("T output contains a constant day")
    return output


# ---- frozen Temporal runtime ----
"""Causal temporal model over the Candidate454 factor panel.

Inputs are daily candidate-factor histories shaped
``[batch_date, stock, lookback, candidate]``.  The model is independent from
the retired synthetic bar-feature layer and from the legacy S/I/T/J pipeline.
"""


from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F



@dataclass(frozen=True)
class CandidateTemporalConfig:
    input_dim: int = 454
    model_dim: int = 128
    lookback: int = 60
    kernels: tuple[int, ...] = (3, 5, 15)
    transformer_layers: int = 2
    attention_heads: int = 8
    feedforward_dim: int = 256
    dropout: float = 0.1
    history_mode: str = "dense"

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.model_dim <= 0 or self.lookback <= 0:
            raise ValueError("input_dim, model_dim, and lookback must be positive")
        if not self.kernels or any(kernel <= 0 for kernel in self.kernels):
            raise ValueError("kernels must contain positive integers")
        if self.model_dim % self.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if self.history_mode not in {"dense", "stride2", "multiscale"}:
            raise ValueError(
                "history_mode must be one of dense, stride2, or multiscale"
            )


def masked_cross_sectional_zscore(
    values: Tensor,
    observed_mask: Tensor,
    stock_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Normalize across stocks separately for every date, lag, and factor."""

    valid = observed_mask.bool() & stock_mask[:, :, None, None].bool() & torch.isfinite(values)
    clean = torch.where(valid, values, torch.zeros_like(values))
    count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    mean = clean.sum(dim=1, keepdim=True) / count
    centered = torch.where(valid, clean - mean, torch.zeros_like(clean))
    variance = centered.square().sum(dim=1, keepdim=True) / count
    scale = variance.sqrt().clamp_min(eps)
    return torch.where(valid, centered / scale, torch.zeros_like(values))


MULTISCALE_HISTORY_LAGS = (0, 1, 2, 3, 4, 6, 9, 14, 19, 29, 44, 59)


def temporal_history_indices(
    length: int,
    mode: str,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Select causal history tokens while always retaining the current date."""

    if length <= 0:
        raise ValueError("history length must be positive")
    if mode == "dense":
        positions = tuple(range(length))
    elif mode == "stride2":
        positions = tuple(range(length - 1, -1, -2))[::-1]
    elif mode == "multiscale":
        positions = tuple(
            sorted({length - 1 - lag for lag in MULTISCALE_HISTORY_LAGS if lag < length})
        )
    else:
        raise ValueError(f"unsupported history mode: {mode}")
    return torch.tensor(positions, dtype=torch.long, device=device)


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class MultiScaleCausalCNN(nn.Module):
    def __init__(self, config: CandidateTemporalConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.branches = nn.ModuleList(
            CausalDepthwiseConv1d(dim, kernel) for kernel in config.kernels
        )
        self.mix = nn.Conv1d(dim * len(config.kernels), dim, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, values: Tensor) -> Tensor:
        channel_first = values.transpose(1, 2)
        encoded = torch.cat([branch(channel_first) for branch in self.branches], dim=1)
        encoded = self.dropout(F.gelu(self.mix(encoded))).transpose(1, 2)
        return self.norm(values + encoded)


class MaskedAttentionPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(model_dim, 1, bias=False)

    def forward(self, values: Tensor, valid_time: Tensor) -> Tensor:
        logits = self.score(values).squeeze(-1).masked_fill(~valid_time, -torch.inf)
        no_history = ~valid_time.any(dim=1)
        logits = torch.where(no_history[:, None], torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1).masked_fill(~valid_time, 0.0)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class TemporalSummary(nn.Module):
    def __init__(self, config: CandidateTemporalConfig) -> None:
        super().__init__()
        self.attention = MaskedAttentionPool(config.model_dim)
        self.merge = nn.Sequential(
            nn.Linear(config.model_dim * 3, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )

    def forward(self, values: Tensor, valid_time: Tensor) -> Tensor:
        count = valid_time.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (values * valid_time.unsqueeze(-1)).sum(dim=1) / count
        positions = torch.arange(values.shape[1], device=values.device)
        last_index = torch.where(valid_time, positions[None, :], -1).amax(dim=1)
        last_index = last_index.clamp_min(0)
        batch_index = torch.arange(values.shape[0], device=values.device)
        last = values[batch_index, last_index]
        attention = self.attention(values, valid_time)
        return self.merge(torch.cat((last, mean, attention), dim=-1))


class DeepSetsContext(nn.Module):
    def __init__(self, config: CandidateTemporalConfig) -> None:
        super().__init__()
        dim = config.model_dim
        self.phi = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.rho = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(dim),
        )

    def forward(self, values: Tensor, stock_mask: Tensor) -> Tensor:
        encoded = self.phi(values)
        valid = stock_mask.unsqueeze(-1).to(encoded.dtype)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        market_mean = (encoded * valid).sum(dim=1, keepdim=True) / count
        centered = (encoded - market_mean) * valid
        market_std = (centered.square().sum(dim=1, keepdim=True) / count).sqrt()
        context = torch.cat(
            (
                encoded,
                market_mean.expand_as(encoded),
                market_std.expand_as(encoded),
                encoded - market_mean,
            ),
            dim=-1,
        )
        return self.rho(context) * valid


class CandidateTemporalNetwork(nn.Module):
    """Candidate-factor temporal encoder with cross-sectional market context."""

    def __init__(self, config: CandidateTemporalConfig | None = None) -> None:
        super().__init__()
        self.config = config or CandidateTemporalConfig()
        cfg = self.config
        self.feature_projection = nn.Sequential(
            nn.Linear(cfg.input_dim * 2, cfg.model_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.model_dim),
        )
        self.cnn = MultiScaleCausalCNN(cfg)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.feedforward_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.temporal_summary = TemporalSummary(cfg)
        self.cross_section = DeepSetsContext(cfg)
        self.head = nn.Sequential(
            nn.Linear(cfg.model_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        values: Tensor,
        observed_mask: Tensor,
        stock_mask: Tensor,
    ) -> Tensor:
        if values.ndim != 4:
            raise ValueError("values must have shape [batch_date, stock, lookback, candidate]")
        batch_size, stock_count, lookback, feature_count = values.shape
        if feature_count != self.config.input_dim or lookback > self.config.lookback:
            raise ValueError("input shape is incompatible with model configuration")
        if observed_mask.shape != values.shape or stock_mask.shape != values.shape[:2]:
            raise ValueError("mask shapes do not match values")

        observed = observed_mask.bool() & torch.isfinite(values)
        normalized = masked_cross_sectional_zscore(values, observed, stock_mask)
        projected = self.feature_projection(
            torch.cat((normalized, observed.to(normalized.dtype)), dim=-1)
        )
        history_indices = temporal_history_indices(
            lookback,
            self.config.history_mode,
            device=values.device,
        )
        projected = projected.index_select(2, history_indices)
        observed = observed.index_select(2, history_indices)
        encoded_lookback = int(history_indices.numel())
        sequence = projected.reshape(batch_size * stock_count, encoded_lookback, -1)
        valid_time = observed.any(dim=-1).reshape(
            batch_size * stock_count,
            encoded_lookback,
        )
        valid_stock = stock_mask.reshape(-1).bool()
        valid_time = valid_time & valid_stock[:, None]
        has_history = valid_time.any(dim=1)

        sequence = self.cnn(sequence)
        causal_mask = torch.triu(
            torch.ones(
                encoded_lookback,
                encoded_lookback,
                device=values.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        safe_padding = ~valid_time
        safe_padding = torch.where(
            has_history[:, None], safe_padding, torch.zeros_like(safe_padding)
        )
        # A left-padded stock can have no admissible key for its earliest
        # causal query.  PyTorch attention then softmaxes an all-masked row and
        # returns NaN, which contaminates the later valid positions.  Keep one
        # explicit missing-data token unmasked; the observed indicators still
        # identify it as missing and TemporalSummary excludes invalid times.
        safe_padding[:, 0] = False
        sequence = self.transformer(
            sequence,
            mask=causal_mask,
            src_key_padding_mask=safe_padding,
        )
        summarized = self.temporal_summary(sequence, valid_time)
        summarized = summarized.reshape(batch_size, stock_count, -1)
        effective_stock_mask = has_history.reshape(batch_size, stock_count)
        contextualized = self.cross_section(summarized, effective_stock_mask)
        # Keep the final cross-sectional projection in fp32.  The temporal
        # backbone may safely use autocast, but trained score differences can
        # be smaller than one fp16 quantization step.  Casting the head to
        # fp16 turned valid rankings into a constant daily cross-section.
        with torch.autocast(device_type=contextualized.device.type, enabled=False):
            scores = self.head(contextualized.float()).squeeze(-1)
        return scores.masked_fill(~effective_stock_mask, 0.0)




# ---- Candidate454 feature runtime ----

import gc
import inspect
from collections.abc import Iterator


BAR_COLUMNS = (
    "date", "instrument", "adjust_factor", "pre_close", "open", "high", "low",
    "close", "deal_number", "volume", "amount",
    *(f"ask_price{i}" for i in range(1, 6)),
    *(f"bid_price{i}" for i in range(1, 6)),
    *(f"ask_volume{i}" for i in range(1, 6)),
    *(f"bid_volume{i}" for i in range(1, 6)),
)
FINANCIAL_COLUMNS = (
    "date", "instrument", "report_date", "shift", "category", "total_assets",
    "total_liabilities", "total_current_assets", "total_current_liabilities",
    "total_owner_equity", "operating_revenue", "operating_profit", "net_profit",
    "net_cffoa", "gross_profit",
)
KEYS = ("date", "instrument")
HISTORY_CALENDAR_DAYS = 183
QUERY_CHUNK_CALENDAR_DAYS = 13
FROZEN_MISSING_INPUTS = (
    # Exact daily cross-sectional residuals require the prohibited industry
    # classification source. The frozen MLP keeps them unobserved via its mask.
    "INT-005", "INT-006", "INT-007", "INT-008", "INT-009", "INT-013",
    # These Fangzheng states require exact free-float turnover not exposed here.
    # financial exposes total/report-period shares, not exact free-float shares.
    "PV-026", "PV-027", "PV-030", "PV-031", "PV-035", "PV-036", "PV-037",
)


def _date_bounds(pd, start_date, end_date):
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date precedes start_date")
    history = start - pd.Timedelta(days=HISTORY_CALENDAR_DAYS)
    return start, end, history


def _load_pool(dai, pd, history, end):
    pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [history.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]},
        compression=True,
    ).df()
    pool["date"] = pd.to_datetime(pool["date"], errors="coerce").dt.normalize()
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.dropna(subset=list(KEYS)).drop_duplicates(list(KEYS))
    pool = pool.loc[pool["date"].between(history, end)]
    return pool.sort_values(list(KEYS)).reset_index(drop=True)


def _load_financial(dai, pd, table, history, end, pool):
    upper = end + pd.Timedelta(days=1)
    sql = f"""
        SELECT {', '.join(FINANCIAL_COLUMNS)}
        FROM {table}
        WHERE date >= TIMESTAMP '{history:%Y-%m-%d}'
          AND date < TIMESTAMP '{upper:%Y-%m-%d}'
        ORDER BY date, instrument, report_date
    """
    frame = dai.query(
        sql,
        filters={"date": [history.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]},
        compression=True,
    ).df()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["disclosure_date"] = frame["date"]
    frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce").dt.normalize()
    frame["instrument"] = frame["instrument"].astype(str)
    calendar = pd.DatetimeIndex(sorted(pool["date"].dropna().unique()))
    positions = calendar.searchsorted(frame["disclosure_date"], side="right")
    valid = positions < len(calendar)
    frame["effective_date"] = pd.NaT
    frame.loc[valid, "effective_date"] = calendar.take(positions[valid]).to_numpy()
    return frame.drop(columns="date")


def _iter_bar1m(dai, pd, table, history, end) -> Iterator[object]:
    cursor = history
    upper_final = end + pd.Timedelta(days=1)
    while cursor < upper_final:
        upper = min(cursor + pd.Timedelta(days=QUERY_CHUNK_CALENDAR_DAYS), upper_final)
        sql = f"""
            SELECT {', '.join(BAR_COLUMNS)}
            FROM {table}
            WHERE date >= TIMESTAMP '{cursor:%Y-%m-%d}'
              AND date < TIMESTAMP '{upper:%Y-%m-%d}'
            ORDER BY date, instrument
        """
        frame = dai.query(
            sql,
            filters={"date": [cursor.strftime("%Y-%m-%d"), upper.strftime("%Y-%m-%d")]},
            compression=True,
        ).df()
        if not frame.empty:
            yield frame
        cursor = upper


def _canonicalize(raw, pool, pl):
    numeric = [column for column in BAR_COLUMNS if column not in {"date", "instrument"}]
    frame = pl.from_pandas(raw, include_index=False, rechunk=True).select(BAR_COLUMNS)
    frame = frame.with_columns(
        pl.col("date").cast(pl.Datetime),
        pl.col("instrument").cast(pl.String),
        *[pl.col(column).cast(pl.Float64, strict=False) for column in numeric],
    ).with_columns(
        pl.col("date").alias("timestamp"),
        pl.col("date").dt.truncate("1d").alias("trade_date"),
        pl.when(pl.col("date").dt.hour() < 12).then(pl.lit("AM")).otherwise(pl.lit("PM")).alias("session_id"),
    )
    pool_pl = pl.from_pandas(pool, include_index=False).with_columns(
        pl.col("date").cast(pl.Datetime), pl.col("instrument").cast(pl.String)
    ).rename({"date": "trade_date"})
    frame = frame.join(pool_pl, on=["trade_date", "instrument"], how="inner")
    frame = frame.drop_nulls(["date", "instrument"]).sort(["instrument", "date"])
    frame = frame.with_columns(
        pl.col("date").cum_count().over(["instrument", "trade_date"]).cast(pl.Int16).alias("minute_index"),
        pl.col("date").cum_count().over(["instrument", "trade_date", "session_id"]).cast(pl.Int16).alias("session_minute_index"),
    )
    return frame


def _daily_bars(frame, pl):
    price = lambda column: (pl.col(column).filter(pl.col(column) > 0))
    daily = frame.group_by(["trade_date", "instrument"], maintain_order=True).agg(
        price("open").first().alias("open_raw"),
        price("high").max().alias("high_raw"),
        price("low").min().alias("low_raw"),
        price("close").last().alias("close_raw"),
        price("pre_close").first().alias("pre_close_raw"),
        pl.col("adjust_factor").drop_nulls().last().alias("adjust_factor"),
        pl.col("amount").filter(pl.col("amount") >= 0).sum().alias("amount"),
        pl.col("volume").filter(pl.col("volume") >= 0).sum().alias("volume"),
        pl.col("deal_number").filter(pl.col("deal_number") >= 0).sum().alias("deal_number"),
        pl.len().cast(pl.Int16).alias("minute_count"),
    ).rename({"trade_date": "date"})
    return daily.with_columns(
        *[(pl.col(column + "_raw") * pl.col("adjust_factor")).cast(pl.Float32).alias(column)
          for column in ("open", "high", "low", "close", "pre_close")],
        (pl.col("amount") / pl.col("volume").replace(0.0, None) * pl.col("adjust_factor")).cast(pl.Float32).alias("vwap"),
    ).select(
        "date", "instrument", "open", "high", "low", "close", "pre_close", "vwap",
        "amount", "volume", "deal_number", "adjust_factor", "minute_count",
    )


def _micro_daily(frame, pl):
    day = ["instrument", "trade_date"]
    session = ["instrument", "trade_date", "session_id"]
    bid_valid = [(pl.col(f"bid_price{i}") > 0) & (pl.col(f"bid_volume{i}") > 0) for i in range(1, 6)]
    ask_valid = [(pl.col(f"ask_price{i}") > 0) & (pl.col(f"ask_volume{i}") > 0) for i in range(1, 6)]
    bid_depth = sum(pl.when(v).then(pl.col(f"bid_volume{i}")).otherwise(0.0) for i, v in enumerate(bid_valid, 1))
    ask_depth = sum(pl.when(v).then(pl.col(f"ask_volume{i}")).otherwise(0.0) for i, v in enumerate(ask_valid, 1))
    previous_close = pl.col("close").shift(1).over(session)
    future5 = pl.col("close").shift(-5).over(session)
    future15 = pl.col("close").shift(-15).over(session)
    frame = frame.with_columns(
        previous_close.alias("_previous_close"), future5.alias("_future5"), future15.alias("_future15"),
        bid_depth.alias("_bid_depth"), ask_depth.alias("_ask_depth"),
        (pl.len().over(day) - pl.col("date").cum_count().over(day) + 1).cast(pl.Int16).alias("_reverse_minute"),
    ).with_columns(
        pl.when((pl.col("close") > 0) & (pl.col("_previous_close") > 0)).then(pl.col("close") / pl.col("_previous_close") - 1).alias("_minute_return"),
        pl.when((pl.col("close") > 0) & (pl.col("_previous_close") > 0)).then((pl.col("close") / pl.col("_previous_close")).log()).alias("_minute_log_return"),
        pl.when(pl.col("close") > 0).then(pl.col("_future5") / pl.col("close") - 1).alias("_future_return5"),
        pl.when(pl.col("close") > 0).then(pl.col("_future15") / pl.col("close") - 1).alias("_future_return15"),
        ((pl.col("amount").clip(lower_bound=0).log1p() + pl.col("volume").clip(lower_bound=0).log1p() + pl.col("deal_number").clip(lower_bound=0).log1p()) / 3).alias("_activity"),
        ((pl.col("ask_price1") + pl.col("bid_price1")) / 2).alias("_mid"),
        ((pl.col("_bid_depth") - pl.col("_ask_depth")) / (pl.col("_bid_depth") + pl.col("_ask_depth")).replace(0.0, None)).alias("_depth_imbalance"),
        ((pl.col("bid_volume1") + pl.col("bid_volume2")) / pl.col("_bid_depth").replace(0.0, None)
         - (pl.col("ask_volume1") + pl.col("ask_volume2")) / pl.col("_ask_depth").replace(0.0, None)).alias("_depth_shape"),
        (sum(v.cast(pl.Int8) for v in bid_valid + ask_valid) / 10.0).alias("_depth_completeness"),
    ).with_columns(
        pl.when((pl.col("bid_price1") > 0) & (pl.col("ask_price1") >= pl.col("bid_price1")) & (pl.col("_mid") > 0))
        .then((pl.col("ask_price1") - pl.col("bid_price1")) / pl.col("_mid")).alias("_relative_spread"),
        pl.when((pl.col("bid_volume1") + pl.col("ask_volume1") > 0) & (pl.col("ask_price1") > pl.col("bid_price1")))
        .then(((pl.col("ask_price1") * pl.col("bid_volume1") + pl.col("bid_price1") * pl.col("ask_volume1"))
               / (pl.col("bid_volume1") + pl.col("ask_volume1")) - pl.col("_mid"))
              / (pl.col("ask_price1") - pl.col("bid_price1"))).alias("_microprice_gap"),
        (pl.col("_mid") / pl.col("_mid").shift(1).over(session) - 1).alias("_mid_return"),
        (pl.col("_bid_depth").shift(-5).over(session) / pl.col("_bid_depth").replace(0.0, None) - 1).alias("_bid_recovery5"),
        (pl.col("_ask_depth").shift(-5).over(session) / pl.col("_ask_depth").replace(0.0, None) - 1).alias("_ask_recovery5"),
    ).with_columns(
        pl.col("_minute_return").abs().quantile(0.90).over(day).alias("_shock90"),
        pl.col("_activity").median().over(day).alias("_activity_median"),
        pl.col("_mid_return").quantile(0.10).over(day).alias("_mid_q10"),
        pl.col("_mid_return").quantile(0.90).over(day).alias("_mid_q90"),
        pl.col("_minute_log_return").std().over(day).alias("_return_sigma"),
    )
    shock = (pl.col("_minute_return").abs() >= pl.col("_shock90")) & (pl.col("_activity") >= pl.col("_activity_median"))
    tail = pl.col("_reverse_minute") <= 60
    valid_quote = pl.col("_mid").is_not_null() & pl.col("_relative_spread").is_not_null()
    bvc = pl.col("volume") * (2 / (1 + (-1.702 * (pl.col("_minute_log_return") / pl.col("_return_sigma")).clip(-6, 6)).exp()) - 1)
    return frame.group_by(["trade_date", "instrument"], maintain_order=True).agg(
        pl.len().cast(pl.Int16).alias("minute_count_micro"),
        pl.col("_minute_log_return").sum().alias("net_log_return"),
        pl.col("_minute_log_return").abs().sum().alias("absolute_log_return"),
        (pl.col("_minute_log_return").pow(2).sum().sqrt()).alias("realized_volatility"),
        (pl.when(pl.col("_minute_log_return") < 0).then(pl.col("_minute_log_return").pow(2)).otherwise(0).sum().sqrt()).alias("downside_realized_volatility"),
        pl.when(tail).then(pl.col("amount")).otherwise(0).sum().alias("tail_60_amount"),
        pl.when(tail).then(pl.col("volume")).otherwise(0).sum().alias("tail_60_volume"),
        pl.when(tail).then(pl.col("deal_number")).otherwise(0).sum().alias("tail_60_deal_number"),
        pl.when(tail).then(pl.col("_minute_log_return")).otherwise(None).sum().alias("tail_60_log_return"),
        pl.when(tail & pl.col("_return_sigma").is_not_null()).then(bvc).otherwise(None).sum().alias("tail_60_signed_volume_bvc"),
        (pl.col("amount").sum() / pl.col("deal_number").sum().replace(0.0, None)).alias("avg_trade_value"),
        (pl.col("volume").sum() / pl.col("deal_number").sum().replace(0.0, None)).alias("avg_trade_volume"),
        (pl.col("_minute_log_return").sum().abs() / pl.col("_minute_log_return").abs().sum().replace(0.0, None)).alias("directional_efficiency"),
        ((pl.when(tail).then(pl.col("amount")).otherwise(0).sum()
          / pl.when(tail).then(pl.col("deal_number")).otherwise(0).sum().replace(0.0, None))
         / (pl.col("amount").sum() / pl.col("deal_number").sum().replace(0.0, None)).replace(0.0, None)).alias("tail_trade_value_ratio"),
        shock.sum().alias("shock_q90_active_count"),
        pl.when(shock & pl.col("_future_return5").is_not_null() & (pl.col("_minute_return") != 0))
        .then((-pl.col("_minute_return").sign() * pl.col("_future_return5") / pl.col("_minute_return").abs()).clip(-2, 2)).otherwise(None).median().alias("shock_q90_recovery_5m_median"),
        valid_quote.sum().alias("valid_snapshot_count"),
        (valid_quote.sum() > 0).alias("micro_snapshot_available"),
        valid_quote.mean().alias("both_sides_valid_rate"),
        pl.all_horizontal([*bid_valid, *ask_valid]).mean().alias("full_five_levels_rate"),
        pl.col("_relative_spread").median().alias("full_day_relative_spread_median"),
        pl.when(tail).then(pl.col("_relative_spread")).otherwise(None).median().alias("tail_60_relative_spread_median"),
        pl.when(tail).then(pl.col("_depth_completeness")).otherwise(None).median().alias("tail_60_depth_completeness_median"),
        pl.col("_depth_imbalance").median().alias("full_day_depth_imbalance_median"),
        pl.col("_depth_imbalance").std().alias("full_day_depth_imbalance_std"),
        pl.when(tail).then(pl.col("_depth_imbalance")).otherwise(None).median().alias("tail_60_bid_depth_imbalance_median"),
        pl.when(tail).then(pl.col("_microprice_gap")).otherwise(None).median().alias("tail_60_microprice_gap_median"),
        pl.when(tail).then(pl.col("_microprice_gap").sign()).otherwise(None).mean().alias("tail_60_microprice_gap_sign_consistency"),
        pl.col("_depth_shape").median().alias("full_day_depth_shape_median"),
        pl.when(tail).then(pl.col("_depth_shape")).otherwise(None).median().alias("tail_60_depth_shape_median"),
        pl.when(tail).then(pl.col("_depth_shape").sign()).otherwise(None).mean().alias("tail_60_shape_sign_consistency"),
        pl.when((pl.col("_mid_return") < 0) & (pl.col("_mid_return") <= pl.col("_mid_q10"))).then(pl.col("_bid_recovery5").clip(-2, 2)).otherwise(None).median().alias("negative_mid_shock_q10_bid_depth_recovery_5m_median"),
        pl.when((pl.col("_mid_return") > 0) & (pl.col("_mid_return") >= pl.col("_mid_q90"))).then(pl.col("_ask_recovery5").clip(-2, 2)).otherwise(None).median().alias("positive_mid_shock_q90_ask_depth_recovery_5m_median"),
    ).rename({"trade_date": "date"})


def _cicc_latent_daily(frame, pl):
    day = ["instrument", "trade_date"]
    session = ["instrument", "trade_date", "session_id"]
    previous = pl.col("close").shift(1).over(session)
    work = frame.with_columns(
        pl.when((pl.col("close") > 0) & (previous > 0)).then((pl.col("close") / previous).log()).alias("_lr"),
        pl.col("volume").shift(-1).over(session).alias("_next_volume"),
        pl.col("volume").sum().over(day).alias("_day_volume"),
        pl.col("date").rank("ordinal").over(day).alias("_minute_index"),
        pl.col("volume").rank("ordinal", descending=True).over(day).alias("_high_rank"),
        pl.col("volume").rank("ordinal").over(day).alias("_low_rank"),
    ).with_columns((pl.col("volume") / pl.col("_day_volume").replace(0.0, None)).alias("_volume_share"))
    positive = pl.col("_lr") > 0
    negative = pl.col("_lr") < 0
    first20 = pl.col("_minute_index") <= 20
    return work.group_by(["trade_date", "instrument"], maintain_order=True).agg(
        pl.col("_lr").filter(pl.col("_high_rank") <= 50).sum().alias("CICC-011"),
        pl.col("_lr").filter(pl.col("_low_rank") <= 50).sum().alias("CICC-012"),
        pl.col("_lr").filter(pl.col("_high_rank") <= 20).sum().alias("CICC-013"),
        pl.col("_lr").filter(pl.col("_low_rank") <= 20).sum().alias("CICC-014"),
        pl.col("_lr").filter(positive).pow(2).sum().sqrt().alias("CICC-018"),
        (pl.col("volume").filter(positive).sum() / pl.col("volume").filter(pl.col("_lr") != 0).sum().replace(0.0, None)).alias("CICC-019"),
        pl.col("_lr").filter(negative).pow(2).sum().sqrt().alias("CICC-020"),
        (pl.col("_lr").skew() / pl.col("_lr").kurtosis().replace(0.0, None)).alias("CICC-024"),
        (pl.col("_volume_share").skew() / pl.col("_volume_share").kurtosis().replace(0.0, None)).alias("CICC-027"),
        pl.corr("close", "_next_volume").alias("CICC-042"),
        (pl.col("_lr") * pl.col("_volume_share")).filter(first20).sum().alias("CICC-076"),
        (-pl.col("_lr").abs() * pl.col("_volume_share")).filter(first20 & negative).sum().alias("CICC-078"),
        (pl.col("_lr") * pl.col("_volume_share")).filter(first20 & positive).sum().alias("CICC-079"),
    ).rename({"trade_date": "date"})


def _changjiang_chunk(canonical, components, pd, np):
    frame = canonical.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["instrument_id"] = frame["instrument"].astype(str)
    frame["session"] = frame["session_id"]
    frame["minute_of_day"] = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame = frame.sort_values(["instrument_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    session_keys = [frame["instrument_id"], frame["date"], frame["session"]]
    frame["ret"] = np.log(frame["close"]).groupby(session_keys, sort=False).diff()
    frame["absret"] = frame["ret"].abs()
    frame["amplitude"] = (frame["high"] - frame["low"]) / frame["open"].where(frame["open"] > 1e-12)
    frame["pvol"] = frame["volume"] / frame["deal_number"].where(frame["deal_number"] > 0)
    frame["pamount"] = frame["amount"] / frame["deal_number"].where(frame["deal_number"] > 0)
    frame["illiq"] = frame["absret"] / frame["amount"].where(frame["amount"] > 0)
    frame["density"] = np.log(frame["absret"].where(frame["absret"] > 1e-12) / frame["volume"].where(frame["volume"] > 0))
    frame["day_position"] = frame.groupby(["instrument_id", "date"], sort=False).cumcount() + 1
    output = components.cj_daily_features(frame, "m1")
    output = output.merge(components.cj_one_minute_extras(frame), on=["date", "instrument_id"], how="left", validate="one_to_one")
    for frequency in (5, 10, 15, 30, 60):
        bars = components.cj_aggregate_bars(frame, frequency)
        part = components.cj_daily_features(bars, f"m{frequency}")
        if frequency >= 10:
            prefix = f"m{frequency}_"
            exact = {"vwret", "ret_std", "first_ret", "first_volume", "rest_equal_ret", "rest_vwret"}
            if frequency == 15:
                exact.update({f"{field}_{metric}" for field in ("volume", "deal", "amp") for metric in ("mean", "std", "cv")})
            if frequency in (10, 30, 60):
                exact.update({f"volume_{side}{threshold:02d}_{metric}" for side, metric in (("low", "invvwret"), ("high", "vwret")) for threshold in (5, 10, 15, 20)})
            keep = ["date", "instrument_id"] + [c for c in part if c.startswith(prefix) and c.removeprefix(prefix) in exact]
            part = part[keep]
        output = output.merge(part, on=["date", "instrument_id"], how="outer", validate="one_to_one")
    return output.rename(columns={"instrument_id": "instrument"})


def _haitong_chunk(canonical, components, pd, np):
    frame = canonical.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["instrument_id"] = frame["instrument"].astype(str)
    frame["session"] = frame["session_id"]
    frame["minute_index"] = frame["session_minute_index"]
    frame = frame.sort_values(["instrument_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    frame["ret"] = np.log(frame["close"]).groupby([frame["instrument_id"], frame["date"], frame["session"]], sort=False).diff()
    one = components.ht_moment_stats(frame, "m1")
    five = components.ht_moment_stats(components.ht_frequency_returns(frame, 5), "m5")
    ten = components.ht_moment_stats(components.ht_frequency_returns(frame, 10), "m10")
    offsets = []
    for offset in range(5):
        part = components.ht_moment_stats(components.ht_frequency_returns(frame, 5, offset=offset), f"offset{offset}")
        part = part.rename(columns={c: c.replace(f"offset{offset}_", "") for c in part if c not in {"date", "instrument_id"}})
        offsets.append(part.assign(offset=offset))
    offset_all = pd.concat(offsets, ignore_index=True)
    m3 = offset_all.groupby(["date", "instrument_id"], sort=False)[["rv_central", "skew_central", "kurt_central"]].mean().add_prefix("m5_all_offsets_").reset_index()
    valid = (frame["bid_price1"] > 0) & (frame["ask_price1"] >= frame["bid_price1"]) & (frame["bid_volume1"] > 0) & (frame["ask_volume1"] > 0)
    frame["l1_strength"] = ((frame["bid_volume1"] - frame["ask_volume1"]) / (frame["bid_volume1"] + frame["ask_volume1"])).where(valid)
    minute = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    frame["tail60_amount"] = frame["amount"].where(minute >= 14 * 60 + 1, 0.0)
    other = frame.groupby(["date", "instrument_id"], sort=False).agg(l1_strength_daily=("l1_strength", "median"), tail60_amount=("tail60_amount", "sum"), total_amount=("amount", "sum"), minute_count=("timestamp", "size")).reset_index()
    output = one
    for part in (five, ten, m3, other):
        output = output.merge(part, on=["date", "instrument_id"], how="outer", validate="one_to_one")
    return output.rename(columns={"instrument_id": "instrument"})


def _join_unique(left, right, pd):
    keys = list(KEYS)
    right = right.copy()
    duplicate = [c for c in right if c not in keys and c in left.columns]
    if duplicate:
        right = right.drop(columns=duplicate)
    return left.merge(right, on=keys, how="left", validate="one_to_one")


def _normalise_components(raw, specs, pd, np):
    keys = raw.loc[:, list(KEYS)].copy()
    data = {}
    for candidate_id, (column, orientation, _group) in specs.items():
        values = pd.to_numeric(raw[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.groupby(raw["date"], sort=False).transform("median")).fillna(0.0)
        rank = values.groupby(raw["date"], sort=False).rank(method="average")
        count = values.groupby(raw["date"], sort=False).transform("count")
        data[candidate_id] = (float(orientation) * 2 * (rank - (count + 1) / 2) / count.where(count > 0)).fillna(0).astype("float32")
    return pd.concat([keys, pd.DataFrame(data, index=raw.index)], axis=1)


def _invoke_direct(direct, financial, daily, pool, pd):
    available = {"financial": financial, "financial_panel": financial, "daily_features": daily,
                 "daily_bars": daily, "bars": daily, "minute_bars": daily, "pool": pool}
    wide = pool.loc[:, list(KEYS)].copy()
    for candidate_id, (builder, _component, _orientation) in direct.CANDIDATE_SPECS.items():
        args = []
        for parameter in inspect.signature(builder).parameters.values():
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD) and parameter.default is inspect.Parameter.empty:
                args.append(available[parameter.name])
        result = builder(*args)
        wide = wide.merge(result.rename(columns={"factor": candidate_id}), on=list(KEYS), how="left", validate="one_to_one")
    return wide


def build_candidate454(
    datasources, start_date, end_date, *, candidate_spec, on_bar1m_chunk=None, return_history=False
):
    import dai
    import numpy as np
    import pandas as pd
    import polars as pl

    from unified_candidate454_runtime import components, direct, gtja
    candidate_ids = tuple(candidate_spec["candidate_ids"])
    component_specs = candidate_spec["component_specs"]
    gtja_specs = candidate_spec["gtja_specs"]

    start, end, history = _date_bounds(pd, start_date, end_date)
    if "bar1m" not in datasources or "financial" not in datasources:
        raise KeyError("datasources must contain bar1m and financial")
    pool = _load_pool(dai, pd, history, end)
    financial = _load_financial(dai, pd, datasources["financial"], history, end, pool)
    daily_parts, micro_parts, cicc_p_parts, cicc_r_parts, cicc_latent_parts = [], [], [], [], []
    fz_parts, cj_parts, ht_parts = [], [], []
    for raw in _iter_bar1m(dai, pd, datasources["bar1m"], history, end):
        raw_start = pd.to_datetime(raw["date"], errors="coerce").min().normalize()
        raw_end = pd.to_datetime(raw["date"], errors="coerce").max().normalize()
        chunk_pool = pool.loc[pool["date"].between(raw_start, raw_end), list(KEYS)]
        canonical_pl = _canonicalize(raw, chunk_pool, pl)
        if canonical_pl.is_empty():
            continue
        if on_bar1m_chunk is not None:
            on_bar1m_chunk(canonical_pl)
        daily_parts.append(_daily_bars(canonical_pl, pl).to_pandas())
        micro_parts.append(_micro_daily(canonical_pl, pl).to_pandas())
        cicc_latent_parts.append(_cicc_latent_daily(canonical_pl, pl).to_pandas())
        canonical = canonical_pl.to_pandas()
        cicc_p_parts.append(components.cicc_compute_pilot_components(canonical))
        cicc_r_parts.append(components.cicc_compute_remaining_components(canonical))
        fz_parts.append(components.fz_compute_minute_daily(canonical))
        cj_parts.append(_changjiang_chunk(canonical, components, pd, np))
        ht_parts.append(_haitong_chunk(canonical, components, pd, np))
        del raw, canonical, canonical_pl
        gc.collect()
    if not daily_parts:
        raise RuntimeError("bar1m scan produced no stock-pool rows")
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(["instrument", "date"]).reset_index(drop=True)
    micro = pd.concat(micro_parts, ignore_index=True).sort_values(["instrument", "date"]).reset_index(drop=True)
    daily["ret"] = daily.groupby("instrument", sort=False)["close"].pct_change(fill_method=None)
    volume = pd.to_numeric(daily["volume"], errors="coerce")
    lagged_volume = volume.groupby(daily["instrument"], sort=False).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    daily["turn"] = volume / lagged_volume.where(lagged_volume > 0)
    daily = _join_unique(daily, micro, pd)

    cicc = pd.concat(cicc_p_parts, ignore_index=True)
    cicc_r = pd.concat(cicc_r_parts, ignore_index=True)
    cicc = _join_unique(cicc, cicc_r, pd)
    cicc = _join_unique(cicc, pd.concat(cicc_latent_parts, ignore_index=True), pd)
    cj_base = _join_unique(daily, pd.concat(cj_parts, ignore_index=True), pd)
    cj_base["first_close"] = pd.to_numeric(cj_base.get("first_close"), errors="coerce") * cj_base["adjust_factor"]
    cj_base["first_open"] = pd.to_numeric(cj_base.get("first_open"), errors="coerce") * cj_base["adjust_factor"]
    cj_panel, _ = components.cj_build_panel(cj_base)
    ht_base = _join_unique(daily, pd.concat(ht_parts, ignore_index=True), pd)
    ht_panel, _ = components.ht_build_factor_panel(ht_base)

    minute_daily = pd.concat(fz_parts, ignore_index=True)
    # These stand-ins are used only so the shared Fangzheng builder can compute
    # report states that do not depend on exposure/turnover.  Every selected
    # candidate that does depend on them is explicitly replaced by NaN below,
    # activating the unchanged checkpoint's native missing-value branches.
    turnover_stub = daily[["date", "instrument", "turn"]].copy()
    classification_stub = daily[["date", "instrument", "amount", "turn"]].copy()
    classification_stub["SIZE"] = np.log1p(
        pd.to_numeric(classification_stub["amount"], errors="coerce").clip(lower=0)
    )
    classification_stub["LIQUIDTY"] = np.log1p(
        pd.to_numeric(classification_stub["turn"], errors="coerce").abs()
    )
    classification_stub["industry_level1_code"] = "__UNAVAILABLE__"
    fz_panel = components.fz_compute_report_factors(
        minute_daily,
        daily[["date", "instrument", "open", "high", "low", "close", "pre_close", "turn"]],
        daily[["date", "instrument", "realized_volatility", "avg_trade_value"]],
        turnover_stub,
        classification_stub[
            ["date", "instrument", "SIZE", "LIQUIDTY", "industry_level1_code"]
        ],
        pool,
    )

    component_raw = pool.loc[:, list(KEYS)].copy()
    for panel in (cicc, cj_panel, ht_panel, fz_panel):
        component_raw = _join_unique(component_raw, panel, pd)
    component_wide = _normalise_components(component_raw, component_specs, pd, np)
    direct_daily = daily
    direct_daily = _join_unique(direct_daily, cicc, pd)
    direct_daily["liq_spread"] = pd.to_numeric(
        direct_daily["full_day_relative_spread_median"], errors="coerce"
    ).where(
        direct_daily["micro_snapshot_available"].fillna(False).astype(bool)
        & pd.to_numeric(direct_daily["valid_snapshot_count"], errors="coerce").ge(30)
    )
    direct_wide = _invoke_direct(direct, financial, direct_daily, pool, pd)

    gtja_panel = pl.from_pandas(daily[["date", "instrument", "open", "high", "low", "close", "pre_close", "amount", "volume"]], include_index=False).rename({"date": "trade_date", "instrument": "stock_code"}).sort(["stock_code", "trade_date"])
    gtja_panel = gtja_panel.with_columns(
        pl.col("open", "high", "low", "close", "pre_close", "amount", "volume")
        .cast(pl.Float64, strict=False)
    )
    gtja_panel = gtja_panel.with_columns(
        (pl.col("close") / pl.col("pre_close") - 1).alias("returns"),
        (pl.col("amount") / pl.col("volume").replace(0.0, None)).alias("vwap"),
        pl.col("amount").rolling_mean(window_size=20, min_samples=5).over("stock_code").alias("cap"),
    )
    gtja_series = []
    for candidate_id, (registry_id, orientation) in gtja_specs.items():
        raw = gtja.REGISTRY[registry_id].impl(gtja_panel)
        scored = gtja_panel.select("trade_date").with_columns(raw.alias("_raw")).with_columns(
            pl.when(pl.col("_raw").is_finite()).then(pl.col("_raw")).otherwise(None).alias("_finite")
        ).with_columns(pl.col("_finite").fill_null(pl.col("_finite").median().over("trade_date")).alias("_filled"))
        scored = scored.with_columns(pl.col("_filled").rank(method="average").over("trade_date").alias("_rank"), pl.col("_filled").count().over("trade_date").alias("_count"))
        gtja_series.append(scored.select((float(orientation) * 2 * (pl.col("_rank") - (pl.col("_count") + 1) / 2) / pl.col("_count")).fill_null(0).fill_nan(0).cast(pl.Float32).alias(candidate_id)).to_series())
    gtja_wide = pd.concat([
        gtja_panel.select(pl.col("trade_date").alias("date"), pl.col("stock_code").alias("instrument")).to_pandas(),
        pl.DataFrame(gtja_series).to_pandas(),
    ], axis=1)
    wide = component_wide.merge(direct_wide, on=list(KEYS), how="left", validate="one_to_one").merge(gtja_wide, on=list(KEYS), how="left", validate="one_to_one")
    missing = sorted(set(candidate_ids).difference(wide.columns))
    if missing:
        raise RuntimeError(f"Candidate454 runtime is missing columns: {missing}")
    wide = wide.loc[:, [*KEYS, *candidate_ids]].sort_values(list(KEYS)).reset_index(drop=True)
    values = wide[list(candidate_ids)].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    available_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in FROZEN_MISSING_INPUTS
    ]
    values[available_ids] = values[available_ids].fillna(0.0).astype("float32")
    values[list(FROZEN_MISSING_INPUTS)] = np.nan
    wide[list(candidate_ids)] = values
    lower = history if return_history else start
    return wide.loc[wide["date"].between(lower, end)].reset_index(drop=True)
