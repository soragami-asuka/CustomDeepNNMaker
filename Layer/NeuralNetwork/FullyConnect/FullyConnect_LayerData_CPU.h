//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
//======================================
#ifndef __FullyConnect_LAYERDATA_CPU_H__
#define __FullyConnect_LAYERDATA_CPU_H__

#include"FullyConnect_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FullyConnect_LayerData_CPU : public FullyConnect_LayerData_Base
	{
		friend class FullyConnect_CPU;

	private:
		// �{��
		std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< �e�j���[�����̌W��<�j���[������, ���͐�> */
		std::vector<NEURON_TYPE>				lpBias;				/**< �j���[�����̃o�C�A�X<�j���[������> */


		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		FullyConnect_LayerData_CPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~FullyConnect_LayerData_CPU();


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
		ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct);
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize);


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
		INNLayer* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif