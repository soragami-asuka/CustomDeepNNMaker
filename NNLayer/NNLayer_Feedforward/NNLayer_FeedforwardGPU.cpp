//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward_DATA.hpp"
#include"NNLayer_FeedforwardBase.h"
#include"NNLayer_Feedforward_FUNC.hpp"

using namespace Gravisbell;
using namespace Gravisbell::NeuralNetwork;




/** CPU�����p�̃��C���[���쐬 */
EXPORT_API Gravisbell::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid)
{
//	return new NNLayer_FeedforwardCPU(guid);
	return NULL;
}