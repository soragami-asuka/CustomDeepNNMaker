//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward.h"
#include"NNLayer_FeedforwardBase.h"





/** CPU�����p�̃��C���[���쐬 */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerGPU(GUID guid)
{
//	return new NNLayer_FeedforwardCPU(guid);
	return NULL;
}