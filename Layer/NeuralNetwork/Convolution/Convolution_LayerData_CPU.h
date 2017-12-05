//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __CONVOLUTION_LAYERDATA_CPU_H__
#define __CONVOLUTION_LAYERDATA_CPU_H__

#include"Convolution_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Convolution_LayerData_CPU : public Convolution_LayerData_Base
	{
		friend class Convolution_CPU;

	private:
		// �{��
		std::vector<NEURON_TYPE>	lpNeuron;			/**< �e�j���[�����̌W�� */
		std::vector<NEURON_TYPE>	lpBias;				/**< �j���[�����̃o�C�A�X<�j���[������> */

		std::vector<NEURON_TYPE*>	lppNeuron;			/**< �e�j���[�����̌W��<�j���[������, ���͐�> */

		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Convolution_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~Convolution_LayerData_CPU();


		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@param	i_config			�ݒ���
			@oaram	i_inputDataStruct	���̓f�[�^�\�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data);
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize );


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
	public:
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif