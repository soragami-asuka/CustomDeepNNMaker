//======================================
// �v�[�����O���C���[�̃f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"MaxAveragePooling_LayerData_CPU.h"
#include"MaxAveragePooling_FUNC.hpp"
#include"MaxAveragePooling_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	MaxAveragePooling_LayerData_CPU::MaxAveragePooling_LayerData_CPU(const Gravisbell::GUID& guid)
		:	MaxAveragePooling_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	MaxAveragePooling_LayerData_CPU::~MaxAveragePooling_LayerData_CPU()
	{
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* MaxAveragePooling_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new MaxAveragePooling_CPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::MaxAveragePooling_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MaxAveragePooling_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data, i_inputDataStruct);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::MaxAveragePooling_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MaxAveragePooling_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	S32 useBufferSize = 0;
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize, useBufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// �g�p�����o�b�t�@�ʂ��i�[
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
