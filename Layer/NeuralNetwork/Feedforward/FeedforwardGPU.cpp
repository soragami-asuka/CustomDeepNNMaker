//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"Feedforward_DATA.hpp"
#include"FeedforwardBase.h"
#include"Feedforward_FUNC.hpp"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;




/** CPU�����p�̃��C���[���쐬 */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid)
{
//	return new FeedforwardCPU(guid);
	return NULL;
}