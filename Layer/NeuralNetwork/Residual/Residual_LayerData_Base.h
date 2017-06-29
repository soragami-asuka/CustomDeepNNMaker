//======================================
// �v�[�����O���C���[�̃f�[�^
//======================================
#ifndef __Residual_DATA_BASE_H__
#define __Residual_DATA_BASE_H__

#include<Layer/ILayerData.h>

#include<vector>

#include"Residual_DATA.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	typedef F32 NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class Residual_LayerData_Base : public ILayerData
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[�f�[�^���ʗp��GUID */

		SettingData::Standard::IData* pLayerStructure;	/**< ���C���[�\�����`�����R���t�B�O�N���X */
//		Residual::LayerStructure layerStructure;	/**< ���C���[�\�� */


		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		Residual_LayerData_Base(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		virtual ~Residual_LayerData_Base();


		//===========================
		// ���ʏ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;

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
		// ���C���[�ݒ�
		//===========================
	public:
		/** �ݒ����ݒ� */
		ErrorCode SetLayerConfig(const SettingData::Standard::IData& config);

		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U32 GetUseBufferByteCount()const;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


	public:
		//===========================
		// ���C���[�\��
		//===========================
		/** ���̓f�[�^�\�����g�p�\���m�F����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	�g�p�\�ȓ��̓f�[�^�\���̏ꍇtrue���Ԃ�. */
		bool CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** �o�̓f�[�^�\�����擾����.
			@param	i_lpInputDataStruct	���̓f�[�^�\���̔z��. GetInputFromLayerCount()�̖߂�l�ȏ�̗v�f�����K�v
			@return	���̓f�[�^�\�����s���ȏꍇ(x=0,y=0,z=0,ch=0)���Ԃ�. */
		IODataStruct GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount);

		/** �����o�͂��\�����m�F���� */
		bool CheckCanHaveMultOutputLayer(void);

		

		//===========================
		// �ŗL�֐�
		//===========================
	public:


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================		
	public:
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]){return ErrorCode::ERROR_CODE_NONE;}
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value){return ErrorCode::ERROR_CODE_NONE;}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value){return ErrorCode::ERROR_CODE_NONE;}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]){return ErrorCode::ERROR_CODE_NONE;}
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif