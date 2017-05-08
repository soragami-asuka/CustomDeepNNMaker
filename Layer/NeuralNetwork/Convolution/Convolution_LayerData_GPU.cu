//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
// GPU����
//======================================
#include"stdafx.h"

#include"Convolution_LayerData_GPU.cuh"
#include"Convolution_FUNC.hpp"
#include"Convolution_GPU.cuh"

#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	Convolution_LayerData_GPU::Convolution_LayerData_GPU(const Gravisbell::GUID& guid)
		:	Convolution_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	Convolution_LayerData_GPU::~Convolution_LayerData_GPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Convolution_LayerData_GPU::Initialize(void)
	{
		// ���̓o�b�t�@�����m�F
		U32 inputBufferCount = this->inputDataStruct.ch * this->layerStructure.FilterSize.z * this->layerStructure.FilterSize.y * this->layerStructure.FilterSize.x;
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �j���[���������m�F
		U32 neuronCount = this->layerStructure.Output_Channel;
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;


		// ��݂��݉񐔂��v�Z
		this->convolutionCount.x = (S32)ceilf((F32)((this->inputDataStruct.x + this->layerStructure.Padding.x*2 - (this->layerStructure.FilterSize.x - 1)) / this->layerStructure.Stride.x));
		this->convolutionCount.y = (S32)ceilf((F32)((this->inputDataStruct.y + this->layerStructure.Padding.y*2 - (this->layerStructure.FilterSize.y - 1)) / this->layerStructure.Stride.y));
		this->convolutionCount.z = (S32)ceilf((F32)((this->inputDataStruct.z + this->layerStructure.Padding.z*2 - (this->layerStructure.FilterSize.z - 1)) / this->layerStructure.Stride.z));

		// ��ݍ��݊J�n�ʒu���v�Z
		this->convolutionStart.x = -(S32)ceilf((F32)(this->layerStructure.Padding.x / this->layerStructure.Stride.x));
		this->convolutionStart.y = -(S32)ceilf((F32)(this->layerStructure.Padding.y / this->layerStructure.Stride.y));
		this->convolutionStart.z = -(S32)ceilf((F32)(this->layerStructure.Padding.z / this->layerStructure.Stride.z));


		// �o�b�t�@���m�ۂ��A�����l��ݒ�
		this->lppNeuron_d.resize(neuronCount * inputBufferCount);
		this->lpBias_d.resize(neuronCount);
		
		thrust::host_vector<F32> lpTmpNeuron(this->lppNeuron_d.size());
		thrust::host_vector<F32> lpTmpBias(this->lpBias_d.size());

		float maxArea = sqrt(6.0f / (inputBufferCount + neuronCount));
		for(U32 i=0; i<lpTmpNeuron.size(); i++)
			lpTmpNeuron[i] = ((F32)Utility::Random::GetValue() - 0.5f) * 2.0f * maxArea;
		for(U32 i=0; i<lpTmpBias.size(); i++)
			lpTmpBias[i] = ((F32)Utility::Random::GetValue() - 0.5f) * 2.0f * maxArea;

		this->lppNeuron_d = lpTmpNeuron;
		this->lpBias_d = lpTmpBias;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode Convolution_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���̓f�[�^�\���̐ݒ�
		this->inputDataStruct = i_inputDataStruct;

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode Convolution_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// �ݒ����ǂݍ���
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(i_lpBuffer, i_bufferSize, readBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ����������
		this->Initialize();

		//// �j���[�����W��
		//for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		//{
		//	memcpy(&this->lppNeuron[neuronNum][0], &i_lpBuffer[readBufferByte], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
		//	readBufferByte += (int)this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		//}

		//// �o�C�A�X
		//memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(NEURON_TYPE));
		//readBufferByte += (int)this->lpBias.size() * sizeof(NEURON_TYPE);


		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Convolution_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		//// �j���[�����W��
		//for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		//{
		//	memcpy(&o_lpBuffer[writeBufferByte], &this->lppNeuron[neuronNum][0], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
		//	writeBufferByte += (int)this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		//}

		//// �o�C�A�X
		//memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(NEURON_TYPE));
		//writeBufferByte += (int)this->lpBias.size() * sizeof(NEURON_TYPE);

		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	INNLayer* Convolution_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new Convolution_GPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// �ǂݎ��Ɏg�p����o�b�t�@�����擾
	Gravisbell::U32 useBufferSize = pLayerData->GetUseBufferByteCount();
	if(useBufferSize >= (Gravisbell::U32)i_bufferSize)
	{
		delete pLayerData;
		return NULL;
	}

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// �g�p�����o�b�t�@�ʂ��i�[
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
